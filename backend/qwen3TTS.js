/**
 * Qwen3-TTS integration for voice cloning and speech synthesis
 * API: https://api.aiinami.com:8443/voice_clone/
 */

import FormData from 'form-data';
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

export class Qwen3TTS {
    constructor() {
        this.baseUrl = process.env.QWEN3_TTS_URL || 'https://api.aiinami.com:8443/voice_clone';
        this.isVoiceRegistered = false;
        this.voiceLanguage = 'ja';

        console.log(`[Qwen3-TTS] Initialized with URL: ${this.baseUrl}`);
    }

    getProviderName() {
        return 'qwen3';
    }

    /**
     * Clone voice from audio data by registering reference audio
     * @param {Buffer} audioBuffer - Audio data (min 5 seconds recommended)
     * @param {string} transcript - Transcription of the audio
     * @param {string} voiceName - Name for the cloned voice (unused, kept for interface compatibility)
     * @param {string} language - Language code ('ja' or 'en')
     * @returns {Promise<string>} - Success indicator
     */
    async cloneVoice(audioBuffer, transcript, voiceName = 'UserClone', language = 'ja') {
        try {
            console.log('[Qwen3-TTS] Starting voice registration...');
            console.log(`[Qwen3-TTS] Audio buffer size: ${audioBuffer.length} bytes`);
            console.log(`[Qwen3-TTS] Transcript: ${transcript.substring(0, 50)}...`);

            // Create form data
            const formData = new FormData();
            formData.append('reference_audio', audioBuffer, {
                filename: 'audio.wav',
                contentType: 'audio/wav'
            });
            formData.append('reference_text', transcript);
            formData.append('language', language === 'ja' ? 'Japanese' : 'English');

            // Register voice
            const response = await fetch(`${this.baseUrl}/register-voice/`, {
                method: 'POST',
                body: formData,
                headers: formData.getHeaders()
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Qwen3-TTS Registration Error (${response.status}): ${errorText}`);
            }

            const result = await response.json();
            console.log('[Qwen3-TTS] Voice registration result:', result);

            this.isVoiceRegistered = true;
            this.voiceLanguage = language;

            console.log(`[Qwen3-TTS] Voice registered successfully. Language: ${language}`);
            return 'registered';

        } catch (error) {
            console.error('[Qwen3-TTS] Voice registration error:', error);
            throw error;
        }
    }

    /**
     * Generate speech from text
     * @param {string} text - Text to synthesize
     * @param {string} voiceId - Voice ID (unused, kept for interface compatibility)
     * @param {string} language - Language code ('ja' or 'en')
     */
    async synthesize(text, voiceId = null, language = null) {
        if (!this.isVoiceRegistered) {
            throw new Error('No reference voice registered. Please clone a voice first.');
        }

        const targetLanguage = language || this.voiceLanguage || 'ja';

        console.log(`[Qwen3-TTS] Synthesizing: "${text}" with language ${targetLanguage}`);

        // Create form data
        const formData = new FormData();
        formData.append('text', text);
        formData.append('language', targetLanguage === 'ja' ? 'Japanese' : 'English');

        // Generate speech with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

        try {
            const response = await fetch(`${this.baseUrl}/generate/`, {
                method: 'POST',
                body: formData,
                headers: formData.getHeaders(),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Qwen3-TTS Synthesis Error (${response.status}): ${errorText}`);
            }

            // Get WAV audio buffer
            const arrayBuffer = await response.arrayBuffer();
            const wavBuffer = Buffer.from(arrayBuffer);

            console.log(`[Qwen3-TTS] Received WAV audio: ${wavBuffer.length} bytes`);

            // Convert WAV to PCM s16le for compatibility with frontend
            const pcmBuffer = this.wavToPcm(wavBuffer);

            console.log(`[Qwen3-TTS] Converted to PCM: ${pcmBuffer.length} bytes`);

            return pcmBuffer;

        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                console.error('[Qwen3-TTS] Request timeout after 30 seconds');
                throw new Error('Qwen3-TTS request timeout. The external API may be slow or unavailable.');
            }
            console.error('[Qwen3-TTS] Synthesis error:', error);
            throw error;
        }
    }

    /**
     * Convert WAV buffer to raw PCM data
     * Assumes WAV is 16-bit PCM
     * @param {Buffer} wavBuffer - WAV file buffer
     * @returns {Buffer} - Raw PCM data
     */
    wavToPcm(wavBuffer) {
        // WAV header is typically 44 bytes for standard PCM
        // Find 'data' chunk
        let dataOffset = 44; // Default offset

        // Try to find 'data' chunk marker
        for (let i = 0; i < Math.min(wavBuffer.length - 4, 100); i++) {
            if (wavBuffer.toString('ascii', i, i + 4) === 'data') {
                // 'data' marker found, data starts after 4-byte size field
                dataOffset = i + 8;
                break;
            }
        }

        // Extract PCM data
        return wavBuffer.slice(dataOffset);
    }

    /**
     * Get current voice ID
     */
    getVoiceId() {
        return this.isVoiceRegistered ? 'registered' : null;
    }

    /**
     * Set voice ID (resets registration status)
     */
    setVoiceId(voiceId) {
        if (voiceId === null) {
            this.isVoiceRegistered = false;
        }
        console.log(`[Qwen3-TTS] Voice registration status: ${this.isVoiceRegistered}`);
    }

    /**
     * Check if Qwen3-TTS service is available
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseUrl}/`, {
                method: 'GET',
                timeout: 5000
            });

            if (response.ok) {
                const data = await response.json();
                console.log('[Qwen3-TTS] Service health:', data);
                return true;
            }
            return false;
        } catch (error) {
            console.error('[Qwen3-TTS] Health check failed:', error.message);
            return false;
        }
    }
}
