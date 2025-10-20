/**
 * Azure OpenAI GPT-4.1-mini integration for advanced next word prediction
 */

import { AzureOpenAI } from 'openai';
import dotenv from 'dotenv';

dotenv.config();

export class AzurePredictor {
  constructor() {
    this.client = new AzureOpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      endpoint: process.env.AZURE_OPENAI_ENDPOINT,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-06-01'
    });

    this.deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT_NAME || 'gpt-4.1-mini';
    this.conversationHistory = [];
  }

  /**
   * Predict next word using GPT-4.1-mini
   * @param {string} context - The current conversation context
   * @param {string} userHistory - User's previous speech for personalization
   * @param {string} language - Language code ('ja' or 'en')
   * @returns {Promise<{word: string, confidence: number, reasoning: string}>}
   */
  async predict(context, userHistory = '', language = 'ja') {
    try {
      const systemPrompt = language === 'ja'
        ? `あなたは日本語音声の次単語予測エンジンです。話者が次に言いそうな自然な単語またはフレーズを予測してください。

ルール:
1. 助詞だけの出力は禁止（が、を、に、へ、で、や、は、も、ので、から、まで、など、か、な、ね、よ、て）
2. 必ず意味のある単語を含めること（名詞、動詞、形容詞など）
3. 1〜3単語程度の自然な続きを予測
4. 直前の文脈と全く同じ単語や表現の繰り返しは避けること
5. 文が完結していたら新しいトピックを予測すること
6. 説明や句読点は不要`
        : `You are an English speech next word prediction engine. Predict the natural word or phrase that the speaker is likely to say next.

Rules:
1. Do not output only articles or prepositions (a, an, the, of, to, in, for, on, at, by, with, etc.)
2. Must include meaningful words (nouns, verbs, adjectives, etc.)
3. Predict 1-3 words as a natural continuation
4. Avoid repeating the exact same words or expressions from the immediate context
5. If the sentence is complete, predict a new topic
6. No explanations or punctuation needed`;

      const userPrompt = language === 'ja'
        ? `履歴: ${userHistory || 'なし'}
文脈: ${context}
話者が次に言いそうな単語:`
        : `History: ${userHistory || 'none'}
Context: ${context}
Next word the speaker is likely to say:`;

      console.log(`[Azure OpenAI] Request to model: ${this.deploymentName}`);
      console.log(`[Azure OpenAI] Context: "${context}"`);

      const response = await this.client.chat.completions.create({
        model: this.deploymentName,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        max_tokens: 10,
        temperature: 0.7,
        top_p: 0.9,
      });

      const predictedText = response.choices[0].message.content.trim();
      console.log(`[Azure OpenAI] Raw response: "${predictedText}"`);

      // For Japanese, don't use \w as it doesn't match Japanese characters
      // Just take the first word/token
      const word = predictedText.split(/\s+/)[0].trim();

      // Check if predicted word is duplicate of last input word
      const lastWords = context.trim().split(/\s+/);
      const lastWord = lastWords[lastWords.length - 1];

      if (word && lastWord && word.includes(lastWord)) {
        console.log(`[Azure OpenAI] Skipping duplicate prediction: "${word}" (matches last word "${lastWord}")`);
        return {
          word: null,
          confidence: 0,
          reasoning: 'duplicate_avoided'
        };
      }

      console.log(`[Azure OpenAI] Predicted word: "${word}" for context: "${context}"`);

      return {
        word: word || null,
        confidence: word ? 0.8 : 0,
        reasoning: 'llm_prediction',
        rawResponse: predictedText
      };

    } catch (error) {
      console.error('[Azure OpenAI] Prediction error:', error.message);
      return {
        word: null,
        confidence: 0,
        reasoning: 'error',
        error: error.message
      };
    }
  }

  /**
   * Add to conversation history for context
   */
  addToHistory(text) {
    this.conversationHistory.push(text);

    // Keep only last 10 utterances to avoid token limits
    if (this.conversationHistory.length > 10) {
      this.conversationHistory.shift();
    }
  }

  /**
   * Get formatted conversation history
   */
  getHistory() {
    return this.conversationHistory.join('\n');
  }

  /**
   * Reset conversation history
   */
  reset() {
    this.conversationHistory = [];
    console.log('[Azure OpenAI] History reset');
  }
}
