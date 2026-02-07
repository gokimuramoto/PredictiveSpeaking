/**
 * PredictiveSpeaking Backend Server
 * Handles real-time voice prediction and synthesis
 */

import express from 'express';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import dotenv from 'dotenv';
import { AzurePredictor } from './azurePredictor.js';
import { CartesiaTTS } from './cartesiaTTS.js';
import { Qwen3TTS } from './qwen3TTS.js';
import { TranscriptCorrector } from './transcriptCorrector.js';
import { RAGPredictor } from './ragPredictor.js';
import { Gpt4oTranscribe } from './gpt4oTranscribe.js';
import { RealtimeTranscribe } from './realtimeTranscribe.js';
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

// Initialize TTS providers
const ttsProviders = {
  cartesia: new CartesiaTTS(),
  qwen3: new Qwen3TTS()
};

// Default provider
let activeTTSProvider = ttsProviders.cartesia;
console.log('🎙️  Default TTS provider: Cartesia');

// Initialize GPT-4o Transcribe ASR (optional)
let gpt4oTranscribe = null;

// Try to connect to GPT-4o Transcribe service
async function initGpt4oTranscribe() {
  try {
    gpt4oTranscribe = new Gpt4oTranscribe();
    const health = await gpt4oTranscribe.checkHealth();
    if (health && health.status === 'healthy') {
      console.log('🎤 GPT-4o Transcribe: Available');
      console.log(`   Model: ${health.model}`);
    } else {
      console.log('⚠️  GPT-4o Transcribe: Not available (will use Browser ASR only)');
      gpt4oTranscribe = null;
    }
  } catch (error) {
    console.log('⚠️  GPT-4o Transcribe: Not available (will use Browser ASR only)');
    gpt4oTranscribe = null;
  }
}

// Call initialization (async, non-blocking)
initGpt4oTranscribe();

// Initialize ASR provider
const ASR_PROVIDER = process.env.ASR_PROVIDER || 'browser';
console.log('🎤 Using Browser Web Speech API');

// Realtime API transcription connections per session
const realtimeConnections = new Map();

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
    ttsProvider: activeTTSProvider.getProviderName(),
    asrProvider: ASR_PROVIDER,
    predictor: 'azure-llm-only',
    voiceId: activeTTSProvider.getVoiceId()
  });
});

// Voice cloning endpoint
app.post('/api/clone-voice', async (req, res) => {
  try {
    let { audioData, transcript, language } = req.body;

    if (!audioData) {
      return res.status(400).json({
        success: false,
        error: 'Missing audioData'
      });
    }

    console.log('[Server] Received voice cloning request');
    console.log(`[Server] Transcript from client: ${transcript || '(empty)'}`);
    console.log(`[Server] Language: ${language || 'ja'}`);

    // Convert base64 to buffer
    const audioBuffer = Buffer.from(audioData.split(',')[1], 'base64');
    console.log(`[Server] Audio buffer size: ${audioBuffer.length} bytes`);

    // If transcript is empty, use GPT-4o to transcribe
    if (!transcript || transcript.trim().length === 0) {
      console.log('[Server] No transcript provided, using GPT-4o to transcribe...');

      if (gpt4oTranscribe) {
        try {
          transcript = await gpt4oTranscribe.transcribe(audioBuffer, language || 'ja');
          console.log(`[Server] GPT-4o transcription result: ${transcript}`);
        } catch (transcribeError) {
          console.error('[Server] GPT-4o transcription failed:', transcribeError.message);
          return res.status(400).json({
            success: false,
            error: 'Transcription failed',
            message: 'Could not transcribe audio. Please try speaking more clearly.'
          });
        }
      } else {
        return res.status(400).json({
          success: false,
          error: 'No transcript and GPT-4o not available',
          message: 'Speech recognition failed and GPT-4o is not available for fallback.'
        });
      }

      if (!transcript || transcript.trim().length === 0) {
        return res.status(400).json({
          success: false,
          error: 'Empty transcript',
          message: 'Could not detect any speech in the recording. Please try again.'
        });
      }
    }

    // Clone voice (pass transcript for providers that need it)
    console.log(`[Server] Attempting voice cloning with ${activeTTSProvider.getProviderName()}...`);
    const voiceId = await activeTTSProvider.cloneVoice(audioBuffer, transcript, 'UserVoice', language || 'ja');
    console.log('[Server] Voice cloning successful!');

    // Reset conversation history before adding new voice clone introduction
    // This ensures old session data doesn't interfere with new voice
    azurePredictor.reset();
    console.log('[Server] Reset conversation history for new voice');

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
  activeTTSProvider.setVoiceId(null);

  res.json({ success: true, message: 'Session reset complete' });
});

// Get available ASR providers
app.get('/api/asr-providers', (req, res) => {
  const hasAzureConfig = !!(process.env.AZURE_OPENAI_ENDPOINT && process.env.AZURE_OPENAI_API_KEY);

  res.json({
    providers: [
      {
        id: 'browser',
        name: 'Browser Web Speech API',
        available: true,
        description: 'Built-in browser speech recognition (Google servers)',
        latency: '~100ms',
        languages: 'Major languages only',
        features: ['Real-time', 'No setup required', 'Online only']
      },
      {
        id: 'realtime',
        name: 'Azure OpenAI Realtime API',
        available: hasAzureConfig,
        description: 'Azure OpenAI streaming transcription via WebSocket',
        latency: '~200ms',
        languages: 'Major languages',
        features: ['Real-time streaming', 'High accuracy', 'Server VAD']
      },
      {
        id: 'gpt4o',
        name: 'GPT-4o-mini Transcribe (Batch)',
        available: gpt4oTranscribe !== null && gpt4oTranscribe.isAvailable,
        description: 'Azure OpenAI batch transcription (2-3 second chunks)',
        latency: '3-5 seconds',
        languages: 'Major languages',
        features: ['High accuracy', 'Fallback option']
      }
    ]
  });
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
          sessionLanguages.set(sessionId, {
            language: data.language || 'ja',
            asrProvider: data.asrProvider || 'browser'
          });
          console.log(`[WebSocket] Session ${sessionId} language set to: ${data.language}`);
          console.log(`[WebSocket] Session ${sessionId} ASR provider: ${data.asrProvider || 'browser'}`);
          break;

        case 'set_asr_provider':
          // Update ASR provider for this session
          const currentConfig = sessionLanguages.get(sessionId) || {};
          sessionLanguages.set(sessionId, {
            language: currentConfig.language || data.language || 'ja',
            asrProvider: data.asrProvider || 'browser'
          });
          console.log(`[WebSocket] Session ${sessionId} ASR provider set to: ${data.asrProvider}`);
          break;

        case 'set_tts_provider':
          if (ttsProviders[data.provider]) {
            activeTTSProvider = ttsProviders[data.provider];
            console.log(`[WebSocket] TTS provider switched to: ${data.provider}`);

            // Notify client of switch
            ws.send(JSON.stringify({
              type: 'tts_provider_changed',
              provider: data.provider
            }));
          } else {
            console.warn(`[WebSocket] Unknown TTS provider requested: ${data.provider}`);
          }
          break;

        case 'start':
          session.isActive = true;
          const sessionConfig = sessionLanguages.get(sessionId) || {};
          ws.send(JSON.stringify({
            type: 'started',
            asrProvider: sessionConfig.asrProvider || ASR_PROVIDER
          }));
          console.log(`[WebSocket] Session ${sessionId} started with ASR: ${sessionConfig.asrProvider || 'browser'}`);
          break;

        case 'stop':
          session.isActive = false;
          ws.send(JSON.stringify({ type: 'stopped' }));
          console.log(`[WebSocket] Session ${sessionId} stopped`);

          // Also stop any Realtime connection
          if (realtimeConnections.has(sessionId)) {
            const rtConn = realtimeConnections.get(sessionId);
            rtConn.disconnect();
            realtimeConnections.delete(sessionId);
            console.log(`[Realtime] Disconnected for session ${sessionId}`);
          }
          break;

        case 'start_realtime':
          // Initialize Realtime API connection for this session
          try {
            const sessionConfig = sessionLanguages.get(sessionId) || {};
            const language = sessionConfig.language || 'ja';

            const rtTranscribe = new RealtimeTranscribe();

            // Set up callbacks
            rtTranscribe.onTranscriptDelta = (delta) => {
              ws.send(JSON.stringify({
                type: 'transcript_delta',
                delta: delta
              }));
            };

            rtTranscribe.onTranscriptCompleted = async (transcript) => {
              // Send transcript to client
              ws.send(JSON.stringify({
                type: 'transcript_update',
                text: transcript
              }));

              // Also handle prediction
              if (transcript && transcript.trim().length > 0) {
                await handleTranscript(sessionId, transcript);
              }
            };

            rtTranscribe.onError = (error) => {
              ws.send(JSON.stringify({
                type: 'error',
                message: 'Realtime transcription error: ' + error.message
              }));
            };

            await rtTranscribe.connect(language);
            realtimeConnections.set(sessionId, rtTranscribe);

            ws.send(JSON.stringify({
              type: 'realtime_connected',
              model: rtTranscribe.model
            }));

            console.log(`[Realtime] Connected for session ${sessionId}, language: ${language}`);
          } catch (error) {
            console.error('[Realtime] Connection error:', error.message);
            ws.send(JSON.stringify({
              type: 'error',
              message: 'Failed to connect to Realtime API: ' + error.message
            }));
          }
          break;

        case 'stop_realtime':
          if (realtimeConnections.has(sessionId)) {
            const rtConn = realtimeConnections.get(sessionId);
            rtConn.disconnect();
            realtimeConnections.delete(sessionId);
            ws.send(JSON.stringify({ type: 'realtime_disconnected' }));
            console.log(`[Realtime] Disconnected for session ${sessionId}`);
          }
          break;

        case 'audio_realtime':
          // Forward audio to Realtime API
          if (realtimeConnections.has(sessionId) && data.audioData) {
            const rtConn = realtimeConnections.get(sessionId);
            const audioBuffer = Buffer.from(data.audioData, 'base64');
            rtConn.sendAudio(audioBuffer);
          }
          break;

        case 'transcript':
          await handleTranscript(sessionId, data.text);
          break;

        case 'audio_gpt4o':
          // Handle audio data from GPT-4o Transcribe mode
          if (gpt4oTranscribe && data.audioData) {
            try {
              const audioBuffer = Buffer.from(data.audioData, 'base64');
              const sessionConfig = sessionLanguages.get(sessionId) || {};
              const language = sessionConfig.language || 'ja';

              console.log(`[GPT-4o Transcribe] Processing audio chunk (${audioBuffer.length} bytes)`);
              const transcript = await gpt4oTranscribe.transcribe(audioBuffer, language);

              if (transcript && transcript.trim().length > 0) {
                console.log(`[GPT-4o Transcribe] Transcript: "${transcript}"`);

                // Send transcript to client for live display
                ws.send(JSON.stringify({
                  type: 'transcript_update',
                  text: transcript
                }));

                await handleTranscript(sessionId, transcript);
              } else {
                console.log('[GPT-4o Transcribe] Empty transcript, skipping');
              }
            } catch (error) {
              console.error('[GPT-4o Transcribe] Error:', error.message);
              ws.send(JSON.stringify({
                type: 'error',
                message: 'GPT-4o Transcribe processing failed: ' + error.message
              }));
            }
          } else if (!gpt4oTranscribe) {
            console.warn('[GPT-4o Transcribe] Audio received but service is not available');
            ws.send(JSON.stringify({
              type: 'error',
              message: 'GPT-4o Transcribe service is not available'
            }));
          }
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
  const sessionConfig = sessionLanguages.get(sessionId) || {};
  const language = sessionConfig.language || 'ja';

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
    transcriptCorrector.corrPredictiveSpeaking/, azurePredictor.getHistory(), language)
      .then(correctedText => {
        if (correctedText !== text) {
          const history = azurePredictor.conversationHistory;
          if (history.length > 0 && history[history.length - 1] === text) {
            history[history.length - 1] = correctedText;
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

    const audioBuffer = await activeTTSProvider.synthesize(predictedWord, null, language);

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
  console.log(`\n🚀 PredictiveSpeaking Server running on http://localhost:${PORT}`);
  console.log(`📊 WebSocket server ready`);
  console.log(`🎙️  Active TTS Provider: ${activeTTSProvider.getProviderName()}`);
  console.log(`🎤 ASR Provider: ${ASR_PROVIDER}`);
  console.log(`🎤 Cartesia API: ${process.env.CARTESIA_API_KEY ? 'Configured' : 'Missing'}`);
  console.log(`🤖 Azure OpenAI: ${process.env.AZURE_OPENAI_API_KEY ? 'Configured' : 'Missing'}`);
  if (ASR_PROVIDER === 'whisper') {
    console.log(`🐳 Whisper Service: ${process.env.WHISPER_SERVICE_URL || 'http://localhost:5000'}`);
  }
  console.log(`\n✨ Ready for voice cloning and prediction!\n`);
});
