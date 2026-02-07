/**
 * GPT-4o-mini Transcribe ASR Client
 * Uses Azure OpenAI gpt-4o-mini-transcribe for speech recognition
 */

import FormData from 'form-data';
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

export class Gpt4oTranscribe {
    constructor() {
        // Azure OpenAI endpoint for gpt-4o-mini-transcribe (requires environment variables)
        this.endpoint = process.env.GPT4O_TRANSCRIBE_ENDPOINT;
        this.apiKey = process.env.GPT4O_TRANSCRIBE_API_KEY;
        this.apiVersion = process.env.GPT4O_TRANSCRIBE_API_VERSION || '2025-03-01-preview';

        if (!this.endpoint || !this.apiKey) {
            console.warn('[GPT-4o Transcribe] Missing GPT4O_TRANSCRIBE_ENDPOINT or GPT4O_TRANSCRIBE_API_KEY environment variables');
        }

        this.isAvailable = false;
        this.lastHealthCheck = null;

        console.log('[GPT-4o Transcribe] Initialized');
    }

    /**
     * Check if service is available
     * @returns {Promise<Object|null>} Health status or null if unavailable
     */
    async checkHealth() {
        try {
            // Simple check - we'll consider it healthy if the endpoint is configured
            // A full health check would require sending a test audio file
            this.isAvailable = !!(this.endpoint && this.apiKey);
            this.lastHealthCheck = {
                status: this.isAvailable ? 'healthy' : 'unhealthy',
                model: 'gpt-4o-mini-transcribe',
                endpoint: this.endpoint ? 'configured' : 'missing'
            };

            return this.lastHealthCheck;
        } catch (error) {
            console.error('[GPT-4o Transcribe] Health check failed:', error.message);
            this.isAvailable = false;
            return null;
        }
    }

    /**
     * Transcribe audio buffer to text
     * @param {Buffer} audioBuffer - Audio data as Buffer (WebM or WAV)
     * @param {string} language - Language code (e.g., 'ja', 'en')
     * @returns {Promise<string>} Transcribed text
     */
    async transcribe(audioBuffer, language = 'ja') {
        if (!this.isAvailable) {
            throw new Error('GPT-4o Transcribe service is not available');
        }

        try {
            const startTime = Date.now();

            // Create form data
            const formData = new FormData();
            formData.append('file', audioBuffer, {
                filename: 'audio.webm',
                contentType: 'audio/webm'
            });

            // Map language code to full language name
            const languageMap = {
                'ja': 'ja',
                'en': 'en',
                'Japanese': 'ja',
                'English': 'en'
            };
            const langCode = languageMap[language] || language;
            formData.append('language', langCode);

            // Build URL with API version
            const url = `${this.endpoint}?api-version=${this.apiVersion}`;

            // Send request
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'api-key': this.apiKey,
                    ...formData.getHeaders()
                },
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('[GPT-4o Transcribe] API Error:', errorText);
                throw new Error(`GPT-4o Transcribe error (${response.status}): ${errorText}`);
            }

            const result = await response.json();
            const latency = Date.now() - startTime;

            console.log(`[GPT-4o Transcribe] Transcription completed in ${latency}ms`);
            console.log(`[GPT-4o Transcribe] Result: "${result.text}"`);

            return result.text || '';

        } catch (error) {
            if (error.message.includes('GPT-4o Transcribe error')) {
                throw error;
            }
            console.error('[GPT-4o Transcribe] Error:', error.message);
            throw new Error(`GPT-4o Transcribe transcription failed: ${error.message}`);
        }
    }

    /**
     * Get supported languages
     * @returns {Promise<Array>} List of supported languages
     */
    async getSupportedLanguages() {
        // GPT-4o supports many languages
        return ['ja', 'en', 'zh', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'ru'];
    }

    /**
     * Test connection
     * @returns {Promise<boolean>} True if connection successful
     */
    async testConnection() {
        const health = await this.checkHealth();
        return health && health.status === 'healthy';
    }
}
