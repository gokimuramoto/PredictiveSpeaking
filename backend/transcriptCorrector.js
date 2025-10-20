/**
 * Transcript Correction Engine
 * 音声認識テキストを文脈に基づいて修正
 */

import { AzureOpenAI } from 'openai';
import dotenv from 'dotenv';

dotenv.config();

export class TranscriptCorrector {
  constructor() {
    this.client = new AzureOpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      endpoint: process.env.AZURE_OPENAI_ENDPOINT,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-06-01'
    });

    this.deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT_NAME || 'gpt-4.1-mini';
  }

  /**
   * 音声認識テキストを修正
   * @param {string} transcript - 音声認識で得られたテキスト
   * @param {string} context - 会話履歴（文脈）
   * @param {string} language - Language code ('ja' or 'en')
   * @returns {Promise<string>} - 修正後のテキスト
   */
  async correct(transcript, context = '', language = 'ja') {
    try {
      const systemPrompt = language === 'ja'
        ? `あなたは音声認識テキストの修正エンジンです。音声認識で得られたテキストを文脈に基づいて修正してください。

ルール:
1. 音声認識の誤り（音が似ている誤変換）を修正
2. 会話履歴と文脈的に自然になるように修正
3. 明らかな誤りがなければ元のテキストをそのまま返す
4. 修正後のテキストのみを出力（説明不要）
5. 過度な修正は避け、音声認識結果を尊重

例:
「きょうわいいてんきですね」→「今日はいい天気ですね」
「それわ違います」→「それは違います」
「こんにちは」→「こんにちは」`
        : `You are a speech recognition text correction engine. Correct the text obtained from speech recognition based on context.

Rules:
1. Fix speech recognition errors (similar-sounding misrecognitions)
2. Make corrections that are natural in the context of the conversation history
3. Return the original text if there are no obvious errors
4. Output only the corrected text (no explanations needed)
5. Avoid excessive corrections and respect the speech recognition results

Examples:
"their going to the store" → "they're going to the store"
"I no that" → "I know that"
"hello there" → "hello there"`;

      const userPrompt = language === 'ja'
        ? `会話履歴:
${context || 'なし'}

音声認識テキスト: ${transcript}

修正後:`
        : `Conversation history:
${context || 'none'}

Speech recognition text: ${transcript}

Corrected:`;

      console.log(`[TranscriptCorrector] Original: "${transcript}"`);

      const response = await this.client.chat.completions.create({
        model: this.deploymentName,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        max_tokens: 100,
        temperature: 0.3,  // 低めの温度で安定した修正
        top_p: 0.9,
      });

      const correctedText = response.choices[0]?.message?.content?.trim() || transcript;

      if (correctedText !== transcript) {
        console.log(`[TranscriptCorrector] Corrected: "${transcript}" → "${correctedText}"`);
      } else {
        console.log(`[TranscriptCorrector] No correction needed`);
      }

      return correctedText;

    } catch (error) {
      console.error('[TranscriptCorrector] Error:', error.message);
      // エラー時は元のテキストを返す
      return transcript;
    }
  }
}
