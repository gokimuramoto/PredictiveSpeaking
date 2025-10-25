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
import { RAGPredictor } from './ragPredictor.js';
import { createServer } from 'http';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.static('frontend'));

// Initialize AI components
const azurePredictor = new AzurePredictor();
const transcriptCorrector = new TranscriptCorrector();
const ragPredictor = new RAGPredictor();

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

    // Add self-introduction transcript to conversation history
    // This allows the LLM to reference the introduction content (name, etc.)
    azurePredictor.addToHistory(transcript);
    console.log('[Server] Added self-introduction to conversation history');

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

// Get available RAG knowledge bases
app.get('/api/rag-knowledge', (req, res) => {
  try {
    const ragDir = path.join(__dirname, '../rag-knowledge');

    if (!fs.existsSync(ragDir)) {
      return res.json({ knowledgeBases: [] });
    }

    const files = fs.readdirSync(ragDir)
      .filter(f => f.endsWith('.json'))
      .map(f => {
        const filePath = path.join(ragDir, f);
        const stats = fs.statSync(filePath);
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));

        return {
          filename: f,
          name: data.modelName || f.replace('.json', ''),
          language: data.language || 'unknown',
          totalChunks: data.stats?.totalChunks || 0,
          avgChunkLength: data.stats?.avgChunkLength || 0,
          createdAt: data.createdAt || stats.mtime.toISOString(),
          sizeKB: (stats.size / 1024).toFixed(2)
        };
      });

    res.json({ models: files });
  } catch (error) {
    console.error('[API] Error listing RAG knowledge bases:', error);
    res.status(500).json({ error: 'Failed to list knowledge bases' });
  }
});

// Load RAG knowledge base
app.post('/api/rag-knowledge/load', async (req, res) => {
  try {
    const { filename } = req.body;

    if (!filename) {
      return res.status(400).json({ error: 'Missing filename' });
    }

    const ragPath = path.join(__dirname, '../rag-knowledge', filename);

    if (!fs.existsSync(ragPath)) {
      return res.status(404).json({ error: 'Knowledge base not found' });
    }

    const success = await ragPredictor.loadKnowledgeBase(ragPath);

    if (success) {
      res.json({
        success: true,
        model: ragPredictor.getModelInfo()
      });
    } else {
      res.status(500).json({ error: 'Failed to load knowledge base' });
    }
  } catch (error) {
    console.error('[API] Error loading RAG knowledge base:', error);
    res.status(500).json({ error: 'Failed to load knowledge base' });
  }
});

// Get current RAG knowledge base info
app.get('/api/rag-knowledge/current', (req, res) => {
  res.json(ragPredictor.getModelInfo());
});

// Unload RAG knowledge base
app.post('/api/rag-knowledge/unload', (req, res) => {
  ragPredictor.unload();
  res.json({ success: true });
});

// Get available folders in knowledge-data directory
app.get('/api/knowledge-folders', (req, res) => {
  try {
    const knowledgeDataDir = path.join(__dirname, '../knowledge-data');

    if (!fs.existsSync(knowledgeDataDir)) {
      return res.json({ folders: [] });
    }

    const entries = fs.readdirSync(knowledgeDataDir, { withFileTypes: true });
    const folders = entries
      .filter(entry => entry.isDirectory())
      .map(entry => ({
        name: entry.name,
        path: `knowledge-data/${entry.name}`
      }));

    res.json({ folders });
  } catch (error) {
    console.error('[API] Error listing knowledge folders:', error);
    res.status(500).json({ error: 'Failed to list folders' });
  }
});

// Build new RAG knowledge base from documents
app.post('/api/rag-knowledge/build', async (req, res) => {
  try {
    const { knowledgeFolder, modelName, language, chunkSize = 500, chunkOverlap = 50 } = req.body;

    if (!knowledgeFolder || !modelName || !language) {
      return res.status(400).json({ error: 'Missing required parameters' });
    }

    console.log('[API] Building RAG knowledge base:', { knowledgeFolder, modelName, language });

    const { spawn } = await import('child_process');
    const buildProcess = spawn('node', [
      path.join(__dirname, 'buildRAG.js'),
      knowledgeFolder,
      modelName,
      language,
      chunkSize.toString(),
      chunkOverlap.toString()
    ], {
      cwd: path.join(__dirname, '..')
    });

    let output = '';
    let errorOutput = '';

    buildProcess.stdout.on('data', (data) => {
      output += data.toString();
      console.log(`[BuildRAG] ${data.toString().trim()}`);
    });

    buildProcess.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error(`[BuildRAG Error] ${data.toString().trim()}`);
    });

    buildProcess.on('close', (code) => {
      if (code === 0) {
        console.log('[API] RAG knowledge base built successfully');
        res.json({
          success: true,
          message: 'Knowledge base built successfully',
          output: output
        });
      } else {
        console.error('[API] RAG knowledge base build failed with code:', code);
        res.status(500).json({
          error: 'Knowledge base build failed',
          message: errorOutput || output,
          exitCode: code
        });
      }
    });

    buildProcess.on('error', (error) => {
      console.error('[API] Error spawning build process:', error);
      res.status(500).json({
        error: 'Failed to start build process',
        message: error.message
      });
    });

  } catch (error) {
    console.error('[API] Error building knowledge base:', error);
    res.status(500).json({ error: 'Failed to build knowledge base', message: error.message });
  }
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
    // Skip prediction if text is empty or only whitespace
    if (!text || text.trim().length === 0) {
      console.log('[Prediction] Skipping empty input');
      return;
    }

    console.log(`[Prediction] Input: "${text}" (language: ${language})`);

    // Update session transcript
    session.transcript += ' ' + text;

    // Transcript correction disabled (was causing unwanted text deletion)
    // If you want to re-enable, uncomment the code below:
    /*
    transcriptCorrector.correct(text, azurePredictor.getHistory(), language)
      .then(correctedText => {
        if (correctedText !== text) {
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
    */

    // Add text to history immediately (without correction)
    azurePredictor.addToHistory(text);

    // Prediction strategy:
    // 1. If RAG loaded: Use LLM with knowledge context (RAG)
    // 2. If RAG not loaded: Use pure LLM
    let predictedWord = null;
    let predictionSource = null;
    let confidence = 0;

    const llmStartTime = Date.now();
    let llmResult = null;

    // Use RAG (LLM + knowledge) if available
    if (ragPredictor.modelLoaded) {
      llmResult = await ragPredictor.predict(text, azurePredictor.getHistory(), language);
      const ragLatency = Date.now() - llmStartTime;

      if (llmResult && llmResult.word) {
        console.log(`[Prediction] RAG took ${ragLatency}ms`);
        console.log(`[Prediction] RAG prediction: "${llmResult.word}" (similarity: ${llmResult.topSimilarity?.toFixed(3)}, chunks: ${llmResult.relevantChunks})`);
        predictedWord = llmResult.word;
        predictionSource = 'rag';
        confidence = llmResult.confidence;
      }
    }

    // Fallback to pure LLM if RAG not loaded or failed
    if (!predictedWord) {
      llmResult = await azurePredictor.predict(text, azurePredictor.getHistory(), language);
      const llmLatency = Date.now() - llmStartTime;

      console.log(`[Prediction] Pure LLM took ${llmLatency}ms`);

      if (!llmResult.word) {
        console.log('[Prediction] No prediction available');
        return;
      }

      predictedWord = llmResult.word;
      predictionSource = 'gpt-4.1-mini';
      confidence = llmResult.confidence;
      console.log(`[Prediction] Pure LLM prediction: "${predictedWord}"`);
    }

    // Send prediction to client immediately (don't wait for TTS)
    ws.send(JSON.stringify({
      type: 'prediction',
      word: predictedWord,
      input: text,  // Include input text for context
      source: predictionSource,
      confidence: confidence
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
