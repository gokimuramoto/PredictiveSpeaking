/**
 * Azure OpenAI Realtime API Transcription Client
 * Uses WebSocket streaming for real-time speech-to-text
 * 
 * Reference: Azure OpenAI Realtime API with intent=transcription
 * URL: wss://{resource}.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&intent=transcription
 */

import WebSocket from 'ws';
import dotenv from 'dotenv';

dotenv.config();

export class RealtimeTranscribe {
    constructor() {
        // Azure OpenAI configuration for gpt-4o-mini-transcribe (East US 2)
        // Credentials must be set in .env file
        const fullEndpoint = process.env.GPT4O_TRANSCRIBE_ENDPOINT || '';
        this.baseEndpoint = fullEndpoint.replace(/\/openai\/deployments.*$/, '');
        this.apiKey = process.env.GPT4O_TRANSCRIBE_API_KEY || '';
        // Realtime API requires 2025-04-01-preview for transcription mode
        this.apiVersion = '2025-04-01-preview';
        // Transcription model
        this.model = 'gpt-4o-mini-transcribe';

        // WebSocket connection
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // Callbacks
        this.onTranscriptDelta = null;  // Called for incremental transcripts
        this.onTranscriptCompleted = null;  // Called when segment is finalized
        this.onError = null;

        // Session language
        this.language = 'ja';

        console.log('[Realtime Transcribe] Initialized with endpoint:', this.baseEndpoint);
    }

    /**
     * Build WebSocket URL for Azure OpenAI Realtime API
     */
    buildWebSocketUrl() {
        // Convert https:// to wss://
        const wsEndpoint = this.baseEndpoint.replace('https://', 'wss://').replace(/\/$/, '');
        return `${wsEndpoint}/openai/realtime?api-version=${this.apiVersion}&intent=transcription`;
    }

    /**
     * Connect to Azure OpenAI Realtime API
     * @param {string} language - Language code ('ja' or 'en')
     */
    async connect(language = 'ja') {
        this.language = language;

        return new Promise((resolve, reject) => {
            try {
                const url = this.buildWebSocketUrl();
                console.log(`[Realtime Transcribe] Connecting to: ${url}`);

                this.ws = new WebSocket(url, {
                    headers: {
                        'api-key': this.apiKey
                    }
                });

                this.ws.on('open', () => {
                    console.log('[Realtime Transcribe] WebSocket connected');
                    this.isConnected = true;
                    this.reconnectAttempts = 0;

                    // Send session configuration
                    this.sendSessionConfig();
                    resolve();
                });

                this.ws.on('message', (data) => {
                    this.handleMessage(data);
                });

                this.ws.on('error', (error) => {
                    console.error('[Realtime Transcribe] WebSocket error:', error.message);
                    if (this.onError) {
                        this.onError(error);
                    }
                    reject(error);
                });

                this.ws.on('close', (code, reason) => {
                    console.log(`[Realtime Transcribe] WebSocket closed: ${code} - ${reason}`);
                    this.isConnected = false;
                    this.handleReconnect();
                });

            } catch (error) {
                console.error('[Realtime Transcribe] Connection error:', error);
                reject(error);
            }
        });
    }

    /**
     * Send session configuration for transcription mode
     */
    sendSessionConfig() {
        const config = {
            type: 'transcription_session.update',
            session: {
                input_audio_format: 'pcm16',
                input_audio_transcription: {
                    model: this.model,
                    language: this.language,
                    prompt: this.language === 'ja'
                        ? '日本語で文字起こしをしてください。必ず日本語のみで出力してください。'
                        : 'Transcribe in English. Output only in English.'
                },
                // Server-side VAD for automatic segment detection
                turn_detection: {
                    type: 'server_vad',
                    threshold: 0.5,
                    prefix_padding_ms: 300,
                    silence_duration_ms: 200
                }
            }
        };

        this.send(config);
        console.log(`[Realtime Transcribe] Session configured for ${this.language}, model: ${this.model}`);
    }

    /**
     * Send audio data to API
     * @param {Buffer} audioBuffer - PCM16 audio data (mono, 24kHz)
     */
    sendAudio(audioBuffer) {
        if (!this.isConnected) {
            console.warn('[Realtime Transcribe] Not connected, skipping audio');
            return;
        }

        const message = {
            type: 'input_audio_buffer.append',
            audio: audioBuffer.toString('base64')
        };

        this.send(message);
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(data) {
        try {
            const message = JSON.parse(data.toString());
            const type = message.type || '';

            switch (type) {
                case 'conversation.item.input_audio_transcription.delta':
                    // Incremental transcript
                    const delta = message.delta || '';
                    if (delta) {
                        console.log(`[Realtime Transcribe] Delta: "${delta}"`);
                        if (this.onTranscriptDelta) {
                            this.onTranscriptDelta(delta);
                        }
                    }
                    break;

                case 'conversation.item.input_audio_transcription.completed':
                    // Final transcript for segment
                    const transcript = message.transcript || '';
                    if (transcript && this.onTranscriptCompleted) {
                        this.onTranscriptCompleted(transcript);
                    }
                    break;

                case 'transcription_session.created':
                case 'transcription_session.updated':
                    console.log(`[Realtime Transcribe] Session ${type.split('.')[1]}`);
                    break;

                case 'error':
                    console.error('[Realtime Transcribe] API error:', message.error);
                    if (this.onError) {
                        this.onError(new Error(message.error?.message || 'Unknown error'));
                    }
                    break;

                default:
                    // Log ALL events for debugging (temporarily verbose)
                    console.log(`[Realtime Transcribe] Event: ${type}`, JSON.stringify(message).substring(0, 200));
            }
        } catch (error) {
            console.error('[Realtime Transcribe] Message parse error:', error);
        }
    }

    /**
     * Handle reconnection logic
     */
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`[Realtime Transcribe] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

            setTimeout(() => {
                this.connect(this.language).catch(err => {
                    console.error('[Realtime Transcribe] Reconnect failed:', err.message);
                });
            }, delay);
        } else {
            console.error('[Realtime Transcribe] Max reconnect attempts reached');
        }
    }

    /**
     * Send message via WebSocket
     */
    send(message) {
        if (this.ws && this.isConnected) {
            this.ws.send(JSON.stringify(message));
        }
    }

    /**
     * Disconnect from API
     */
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
        }
        console.log('[Realtime Transcribe] Disconnected');
    }

    /**
     * Check if connected
     */
    isReady() {
        return this.isConnected;
    }

    /**
     * Health check
     */
    async checkHealth() {
        return {
            status: this.isConnected ? 'connected' : 'disconnected',
            model: this.model,
            endpoint: this.endpoint ? 'configured' : 'missing'
        };
    }
}
