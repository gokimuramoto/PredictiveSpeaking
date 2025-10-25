/**
 * RAG-based Predictor using knowledge data
 * Performs semantic search to find relevant context for prediction
 */

import fs from 'fs';
import path from 'path';
import { AzureOpenAI } from 'openai';
import dotenv from 'dotenv';

dotenv.config();

export class RAGPredictor {
  constructor() {
    this.client = new AzureOpenAI({
      apiKey: process.env.AZURE_OPENAI_API_KEY,
      endpoint: process.env.AZURE_OPENAI_ENDPOINT,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-06-01'
    });

    this.deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT_NAME || 'gpt-4.1-mini';
    this.embeddingModel = process.env.AZURE_OPENAI_EMBEDDING_MODEL || 'text-embedding-ada-002';

    this.knowledgeBase = []; // Array of {text, embedding}
    this.modelLoaded = false;
    this.modelName = null;
    this.language = 'ja';
  }

  /**
   * Load knowledge base from JSON file
   */
  async loadKnowledgeBase(knowledgeBasePath) {
    try {
      console.log(`[RAG] Loading knowledge base from: ${knowledgeBasePath}`);

      if (!fs.existsSync(knowledgeBasePath)) {
        console.warn(`[RAG] Knowledge base file not found: ${knowledgeBasePath}`);
        return false;
      }

      const data = JSON.parse(fs.readFileSync(knowledgeBasePath, 'utf-8'));

      this.knowledgeBase = data.chunks || [];
      this.modelName = data.modelName;
      this.language = data.language || 'ja';

      console.log(`[RAG] Knowledge base loaded successfully`);
      console.log(`[RAG] - Model name: ${this.modelName}`);
      console.log(`[RAG] - Language: ${this.language}`);
      console.log(`[RAG] - Total chunks: ${this.knowledgeBase.length}`);

      this.modelLoaded = true;
      return true;
    } catch (error) {
      console.error('[RAG] Error loading knowledge base:', error);
      this.modelLoaded = false;
      return false;
    }
  }

  /**
   * Create embedding for text using Azure OpenAI
   */
  async createEmbedding(text) {
    try {
      const response = await this.client.embeddings.create({
        model: this.embeddingModel,
        input: text
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error('[RAG] Error creating embedding:', error.message);
      return null;
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Search for relevant chunks using semantic similarity
   */
  async searchRelevantChunks(query, topK = 3) {
    if (!this.modelLoaded || this.knowledgeBase.length === 0) {
      return [];
    }

    // Create embedding for query
    const queryEmbedding = await this.createEmbedding(query);
    if (!queryEmbedding) return [];

    // Calculate similarity scores
    const scores = this.knowledgeBase.map(chunk => ({
      text: chunk.text,
      score: this.cosineSimilarity(queryEmbedding, chunk.embedding)
    }));

    // Sort by similarity and return top K
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }

  /**
   * Predict next word using RAG
   */
  async predict(context, conversationHistory = '', language = 'ja') {
    if (!this.modelLoaded) {
      return null;
    }

    try {
      const startTime = Date.now();

      // Search for relevant knowledge
      const relevantChunks = await this.searchRelevantChunks(context, 3);

      if (relevantChunks.length === 0 || relevantChunks[0].score < 0.3) {
        console.log('[RAG] No relevant knowledge found (low similarity)');
        return null;
      }

      const searchTime = Date.now() - startTime;
      console.log(`[RAG] Found ${relevantChunks.length} relevant chunks in ${searchTime}ms`);
      console.log(`[RAG] Top similarity: ${relevantChunks[0].score.toFixed(3)}`);

      // Build context from relevant chunks
      const knowledgeContext = relevantChunks
        .map((chunk, i) => `[関連知識${i + 1}] ${chunk.text}`)
        .join('\n\n');

      // Generate prediction with RAG context
      const systemPrompt = language === 'ja'
        ? `あなたは専門知識に基づいて次の単語を予測するエンジンです。

以下の関連知識を参考にして、話者が次に言いそうな自然な単語またはフレーズを予測してください。

${knowledgeContext}

ルール:
1. 関連知識の内容を活用すること
2. 助詞だけの出力は禁止
3. 1〜3単語程度の自然な続きを予測
4. 説明や句読点は不要`
        : `You are a prediction engine based on specialized knowledge.

Use the following relevant knowledge to predict the natural word or phrase the speaker is likely to say next.

${knowledgeContext}

Rules:
1. Utilize the content from relevant knowledge
2. Do not output only articles or prepositions
3. Predict 1-3 words as a natural continuation
4. No explanations or punctuation needed`;

      const userPrompt = language === 'ja'
        ? `文脈: ${context}\n話者が次に言いそうな単語:`
        : `Context: ${context}\nNext word the speaker is likely to say:`;

      const llmStartTime = Date.now();
      const response = await this.client.chat.completions.create({
        model: this.deploymentName,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt }
        ],
        max_tokens: 20,
        temperature: 0.7,
        top_p: 0.9,
      });

      const predictedText = response.choices[0].message.content.trim();
      const llmTime = Date.now() - llmStartTime;
      const totalTime = Date.now() - startTime;

      console.log(`[RAG] LLM prediction: "${predictedText}" (LLM: ${llmTime}ms, Total: ${totalTime}ms)`);

      // Extract word (similar to azurePredictor)
      const word = language === 'ja'
        ? predictedText.split(/\s+/)[0].trim()
        : predictedText.replace(/[。、.,!?;:"'「」『』（）()[\]]+$/g, '').trim();

      return {
        word: word || null,
        confidence: relevantChunks[0].score,
        reasoning: 'rag_prediction',
        rawResponse: predictedText,
        relevantChunks: relevantChunks.length,
        topSimilarity: relevantChunks[0].score
      };

    } catch (error) {
      console.error('[RAG] Prediction error:', error.message);
      return null;
    }
  }

  /**
   * Get model information
   */
  getModelInfo() {
    return {
      loaded: this.modelLoaded,
      modelName: this.modelName,
      language: this.language,
      totalChunks: this.knowledgeBase.length,
      type: 'rag'
    };
  }

  /**
   * Unload current knowledge base
   */
  unload() {
    this.knowledgeBase = [];
    this.modelLoaded = false;
    this.modelName = null;
    console.log('[RAG] Knowledge base unloaded');
  }
}
