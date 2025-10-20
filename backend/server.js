/**
 * EchoNext Backend Server
 * Handles real-time voice prediction and synthesis
 */

import express from 'express';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import dotenv from 'dotenv';
import { AzurePredictor } from './azurePredictor.js';
import { CartesiaTTS } from './cartesiaTTS.js';
import { TranscriptCorrector } from './transcriptCorrector.js';
import { createServer } from 'http';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('frontend'));

// Initialize AI components (LLM-only, no N-gram)
const azurePredictor = new AzurePredictor();
const transcriptCorrector = new TranscriptCorrector();

// Initialize TTS provider (Cartesia only now)
console.log('🎙️  Using Cartesia TTS provider');
const ttsProvider = new CartesiaTTS();

// Initialize ASR provider (browser only - whisper removed)
const ASR_PROVIDER = process.env.ASR_PROVIDER || 'browser';

if (ASR_PROVIDER === 'whisper') {
  console.warn('⚠️  Whisper ASR not available in this version. Falling back to Browser Web Speech API.');
  console.log('🎤 Using Browser Web Speech API');
} else {
  console.log('🎤 Using Browser Web Speech API');
}

// Session state
const sessions = new Map();

// Language settings per session
const sessionLanguages = new Map();

// Create HTTP server
const server = createServer(app);

// WebSocket server
const wss = new WebSocketServer({ server });

// REST API Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    ttsProvider: ttsProvider.getProviderName(),
    asrProvider: ASR_PROVIDER,
    predictor: 'azure-llm-only',
    voiceId: ttsProvider.getVoiceId()
  });
});

// Voice cloning endpoint
app.post('/api/clone-voice', async (req, res) => {
  try {
    const { audioData, transcript, language } = req.body;

    if (!audioData || !transcript) {
      return res.status(400).json({
        success: false,
        error: 'Missing audioData or transcript'
      });
    }

    console.log('[Server] Received voice cloning request');
    console.log(`[Server] Transcript: ${transcript}`);
    console.log(`[Server] Language: ${language || 'ja'}`);

    // Convert base64 to buffer
    const audioBuffer = Buffer.from(audioData.split(',')[1], 'base64');
    console.log(`[Server] Audio buffer size: ${audioBuffer.length} bytes`);

    // Clone voice (pass transcript for providers that need it)
    console.log('[Server] Attempting voice cloning...');
    const voiceId = await ttsProvider.cloneVoice(audioBuffer, transcript, 'UserVoice', language || 'ja');
    console.log('[Server] Voice cloning successful!');

    // Do NOT train on voice cloning transcript
    // We only want to learn from real-time speech during active system
    console.log('[Server] Voice cloning complete (training skipped)');

    res.json({
      success: true,
      voiceId: voiceId
    });

  } catch (error) {
    console.error('[Server] Voice cloning error:', error);

    // Return detailed error information
    res.status(500).json({
      success: false,
      error: 'Voice cloning failed',
      message: error.message,
      details: error.cause?.message || 'Unknown error occurred'
    });
  }
});

// Reset session endpoint
app.post('/api/reset', (req, res) => {
  console.log('[Server] Resetting session...');

  azurePredictor.reset();
  ttsProvider.setVoiceId(null);

  res.json({ success: true, message: 'Session reset complete' });
});

// WebSocket connection handler
wss.on('connection', (ws) => {
  const sessionId = Date.now().toString();
  sessions.set(sessionId, {
    ws: ws,
    isActive: false,
    transcript: ''
  });

  console.log(`[WebSocket] Client connected. Session: ${sessionId}`);

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message.toString());
      const session = sessions.get(sessionId);

      switch (data.type) {
        case 'set_language':
          // Store language preference for this session
          sessionLanguages.set(sessionId, data.language || 'ja');
          console.log(`[WebSocket] Session ${sessionId} language set to: ${data.language}`);
          break;

        case 'start':
          session.isActive = true;
          ws.send(JSON.stringify({
            type: 'started',
            asrProvider: ASR_PROVIDER
          }));
          console.log(`[WebSocket] Session ${sessionId} started`);
          break;

        case 'stop':
          session.isActive = false;
          ws.send(JSON.stringify({ type: 'stopped' }));
          console.log(`[WebSocket] Session ${sessionId} stopped`);
          break;

        case 'transcript':
          await handleTranscript(sessionId, data.text);
          break;

        case 'audio':
          // Handle audio data from frontend (whisper not available in this version)
          console.warn('[WebSocket] Audio message received but Whisper ASR is not available');
          break;

        default:
          console.warn(`[WebSocket] Unknown message type: ${data.type}`);
      }

    } catch (error) {
      console.error('[WebSocket] Message handling error:', error);
      ws.send(JSON.stringify({
        type: 'error',
        message: error.message
      }));
    }
  });

  ws.on('close', () => {
    sessions.delete(sessionId);
    sessionLanguages.delete(sessionId);
    console.log(`[WebSocket] Client disconnected. Session: ${sessionId}`);
  });
});

/**
 * Handle incoming transcript and generate prediction + TTS
 */
async function handleTranscript(sessionId, text) {
  const session = sessions.get(sessionId);

  if (!session || !session.isActive) {
    return;
  }

  const ws = session.ws;
  const language = sessionLanguages.get(sessionId) || 'ja';

  try {
    console.log(`[Prediction] Input: "${text}" (language: ${language})`);

    // Update session transcript
    session.transcript += ' ' + text;

    // 非同期でテキスト修正を実行（リアルタイム予測を妨げない）
    // 修正後のテキストで履歴を更新
    transcriptCorrector.correct(text, azurePredictor.getHistory(), language)
      .then(correctedText => {
        if (correctedText !== text) {
          // 履歴の最後の項目を修正後のテキストで置き換え
          const history = azurePredictor.conversationHistory;
          if (history.length > 0 && history[history.length - 1] === text) {
            history[history.length - 1] = correctedText;
            console.log(`[History] Updated with corrected text: "${correctedText}"`);
          }
        }
      })
      .catch(err => {
        console.error('[TranscriptCorrector] Background correction failed:', err.message);
      });

    // Add original text to history immediately (for real-time prediction)
    azurePredictor.addToHistory(text);

    // Use LLM for prediction (no N-gram)
    const llmStartTime = Date.now();
    const azureResult = await azurePredictor.predict(text, azurePredictor.getHistory(), language);
    const llmLatency = Date.now() - llmStartTime;

    console.log(`[Prediction] LLM took ${llmLatency}ms`);

    if (!azureResult.word) {
      console.log('[Prediction] No prediction available');
      return;
    }

    const predictedWord = azureResult.word;
    console.log(`[Prediction] Predicted: "${predictedWord}"`);

    // Send prediction to client immediately (don't wait for TTS)
    ws.send(JSON.stringify({
      type: 'prediction',
      word: predictedWord,
      input: text,  // Include input text for context
      source: 'gpt-4.1-mini',
      confidence: azureResult.confidence
    }));

    // Generate TTS for predicted word
    console.log(`[TTS] Synthesizing: "${predictedWord}"`);
    const ttsStartTime = Date.now();

    const audioBuffer = await ttsProvider.synthesize(predictedWord, null, language);

    const ttsLatency = Date.now() - ttsStartTime;
    console.log(`[TTS] Synthesis took ${ttsLatency}ms`);

    // Send audio to client
    ws.send(JSON.stringify({
      type: 'audio',
      word: predictedWord,
      audio: audioBuffer.toString('base64'),
      format: 'pcm_s16le',
      sampleRate: 22050
    }));

    console.log(`[TTS] Audio sent for word: "${predictedWord}"`);

  } catch (error) {
    console.error('[Prediction] Error:', error);
    // Don't send WebSocket connection errors to client
    if (!error.message.includes('Not connected to WebSocket')) {
      ws.send(JSON.stringify({
        type: 'error',
        message: error.message
      }));
    }
  }
}

// Start server
server.listen(PORT, () => {
  console.log(`\n🚀 EchoNext Server running on http://localhost:${PORT}`);
  console.log(`📊 WebSocket server ready`);
  console.log(`🎙️  TTS Provider: ${ttsProvider.getProviderName()}`);
  console.log(`🎤 ASR Provider: ${ASR_PROVIDER}`);
  console.log(`🎤 Cartesia API: ${process.env.CARTESIA_API_KEY ? 'Configured' : 'Missing'}`);
  console.log(`🤖 Azure OpenAI: ${process.env.AZURE_OPENAI_API_KEY ? 'Configured' : 'Missing'}`);
  if (ASR_PROVIDER === 'whisper') {
    console.log(`🐳 Whisper Service: ${process.env.WHISPER_SERVICE_URL || 'http://localhost:5000'}`);
  }
  console.log(`\n✨ Ready for voice cloning and prediction!\n`);
});
