/**
 * Cartesia TTS integration for voice cloning and speech synthesis
 */

import Cartesia from '@cartesia/cartesia-js';
import dotenv from 'dotenv';

dotenv.config();

export class CartesiaTTS {
  constructor() {
    this.client = new Cartesia({
      apiKey: process.env.CARTESIA_API_KEY
    });

    this.voiceId = null; // Will be set after voice cloning
    this.defaultVoiceId = 'a0e99841-438c-4a64-b679-ae501e7d6091'; // Cartesia default voice
  }

  getProviderName() {
    return 'cartesia';
  }

  /**
   * Clone voice from audio data
   * @param {Buffer} audioBuffer - Audio data (min 5 seconds recommended)
   * @param {string} transcript - Transcription of the audio (not used by Cartesia)
   * @param {string} voiceName - Name for the cloned voice
   * @param {string} language - Language code ('ja' or 'en')
   * @returns {Promise<string>} - Voice embedding (to use with mode: "embedding")
   */
  async cloneVoice(audioBuffer, transcript, voiceName = 'UserClone', language = 'ja') {
    try {
      console.log('[Cartesia] Starting voice cloning...');
      console.log(`[Cartesia] Audio buffer size: ${audioBuffer.length} bytes`);

      // Cartesia expects a Blob object (not File in Node.js)
      const blob = new Blob([audioBuffer], { type: 'audio/webm' });

      console.log('[Cartesia] Sending clone request to API...');

      // Clone voice using raw fetch to get better error messages
      let embedding;

      // Make direct API call to get proper error messages
      const formData = new FormData();
      formData.append('clip', blob, 'voice.webm');
      formData.append('mode', 'similarity');  // Use similarity mode (stability removed in 2025-04-16)

      try {
        const response = await fetch('https://api.cartesia.ai/voices/clone/clip', {
          method: 'POST',
          headers: {
            'X-API-Key': process.env.CARTESIA_API_KEY,
            'Cartesia-Version': '2024-11-13'  // Use 2024-11-13 for stable cloning support
          },
          body: formData
        });

        console.log('[Cartesia] API Response Status:', response.status, response.statusText);

        // Get response text first
        const responseText = await response.text();
        console.log('[Cartesia] API Response Body:', responseText);

        if (!response.ok) {
          throw new Error(`Cartesia API Error (${response.status}): ${responseText}`);
        }

        // Try to parse as JSON
        const result = JSON.parse(responseText);
        embedding = result.embedding;

      } catch (apiError) {
        console.error('[Cartesia] API Error:', apiError.message);
        throw new Error(`Voice cloning failed: ${apiError.message}`);
      }

      console.log('[Cartesia] Clone API response received');
      console.log(`[Cartesia] Embedding length: ${embedding.length}`);

      // Store the embedding
      this.voiceEmbedding = embedding;

      console.log('[Cartesia] Creating permanent voice...');

      // Optionally, create a permanent voice with this embedding
      const createdVoice = await this.client.voices.create({
        name: voiceName,
        description: 'User voice clone for digital twin',
        embedding: embedding
      });

      this.voiceId = createdVoice.id;
      this.voiceLanguage = language; // Store language for this voice
      console.log(`[Cartesia] Voice cloned successfully. Voice ID: ${this.voiceId}, Language: ${language}`);

      return this.voiceId;

    } catch (error) {
      console.error('[Cartesia] Voice cloning error:', error);
      console.error('[Cartesia] Error details:', error.cause || error.message);
      throw error;
    }
  }

  /**
   * Generate speech from text using REST API
   * @param {string} text - Text to synthesize
   * @param {string} voiceId - Voice ID (optional, uses cloned voice if available)
   * @param {string} language - Language code (optional)
   * @returns {Promise<Buffer>} - Audio data
   */
  async synthesize(text, voiceId = null, language = null) {
    return this.synthesizeQuick(text, voiceId, language);
  }

  /**
   * Internal synthesis implementation using REST API
   */
  async synthesizeQuick(text, voiceId = null, language = null) {
    try {
      const targetVoiceId = voiceId || this.voiceId || this.defaultVoiceId;
      const targetLanguage = language || this.voiceLanguage || 'ja';

      console.log(`[Cartesia] Quick synthesizing: "${text}" with voice ${targetVoiceId}, language: ${targetLanguage}`);

      // Use direct REST API call for reliability
      const response = await fetch('https://api.cartesia.ai/tts/bytes', {
        method: 'POST',
        headers: {
          'X-API-Key': process.env.CARTESIA_API_KEY,
          'Cartesia-Version': '2024-11-13',  // Match clone API version
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model_id: 'sonic-multilingual',
          transcript: text,
          voice: {
            mode: 'id',
            id: targetVoiceId
          },
          output_format: {
            container: 'raw',
            encoding: 'pcm_s16le',
            sample_rate: 22050  // Reduced for lower latency
          },
          language: targetLanguage,  // Use selected language
          speed: 'normal',  // Use normal speed for better pronunciation accuracy
          add_timestamps: false
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Cartesia TTS API Error (${response.status}): ${errorText}`);
      }

      // Get audio buffer
      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = Buffer.from(arrayBuffer);

      console.log(`[Cartesia] Quick synthesis complete. Audio size: ${audioBuffer.length} bytes`);

      return audioBuffer;

    } catch (error) {
      console.error('[Cartesia] Quick synthesis error:', error);
      throw error;
    }
  }

  /**
   * Get current voice ID
   */
  getVoiceId() {
    return this.voiceId;
  }

  /**
   * Set voice ID manually
   */
  setVoiceId(voiceId) {
    this.voiceId = voiceId;
    console.log(`[Cartesia] Voice ID set to: ${voiceId}`);
  }
}
