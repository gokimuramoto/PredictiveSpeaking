/**
 * Build RAG knowledge base from documents
 * Usage: node backend/buildRAG.js <input-folder> <output-name> <language>
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { PDFExtract } from 'pdf.js-extract';
import mammoth from 'mammoth';
import { AzureOpenAI } from 'openai';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const pdfExtract = new PDFExtract();

// Initialize Azure OpenAI client
const client = new AzureOpenAI({
  apiKey: process.env.AZURE_OPENAI_API_KEY,
  endpoint: process.env.AZURE_OPENAI_ENDPOINT,
  apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-06-01'
});

const embeddingModel = process.env.AZURE_OPENAI_EMBEDDING_MODEL || 'text-embedding-ada-002';

class RAGBuilder {
  constructor(language = 'ja', chunkSize = 500, chunkOverlap = 50) {
    this.language = language;
    this.chunkSize = chunkSize; // Characters per chunk
    this.chunkOverlap = chunkOverlap; // Overlap between chunks
    this.chunks = [];
  }

  /**
   * Split text into overlapping chunks
   */
  chunkText(text) {
    const chunks = [];
    const sentences = this.language === 'ja'
      ? text.split(/[。！？\n]+/).filter(s => s.trim().length > 0)
      : text.split(/[.!?\n]+/).filter(s => s.trim().length > 0);

    let currentChunk = '';

    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (!trimmed) continue;

      if ((currentChunk + trimmed).length > this.chunkSize && currentChunk.length > 0) {
        chunks.push(currentChunk.trim());

        // Add overlap from end of previous chunk
        const words = currentChunk.split(/\s+/);
        const overlapWords = words.slice(-Math.floor(this.chunkOverlap / 10));
        currentChunk = overlapWords.join(' ') + ' ' + trimmed;
      } else {
        currentChunk += (currentChunk ? ' ' : '') + trimmed;
      }
    }

    if (currentChunk.trim().length > 0) {
      chunks.push(currentChunk.trim());
    }

    return chunks;
  }

  /**
   * Create embedding for text
   */
  async createEmbedding(text) {
    try {
      const response = await client.embeddings.create({
        model: embeddingModel,
        input: text
      });

      return response.data[0].embedding;
    } catch (error) {
      console.error('[RAG Builder] Error creating embedding:', error.message);
      throw error;
    }
  }

  /**
   * Process text and create embeddings
   */
  async processText(text) {
    const textChunks = this.chunkText(text);
    console.log(`[RAG Builder] Created ${textChunks.length} chunks`);

    let processedCount = 0;

    for (const chunk of textChunks) {
      try {
        const embedding = await this.createEmbedding(chunk);

        this.chunks.push({
          text: chunk,
          embedding: embedding
        });

        processedCount++;
        if (processedCount % 10 === 0) {
          console.log(`[RAG Builder] Processed ${processedCount}/${textChunks.length} chunks...`);
        }

        // Rate limiting: 60 requests/minute for embeddings
        await new Promise(resolve => setTimeout(resolve, 1100));

      } catch (error) {
        console.error(`[RAG Builder] Error processing chunk: ${error.message}`);
      }
    }

    console.log(`[RAG Builder] Successfully processed ${processedCount} chunks`);
  }

  /**
   * Save knowledge base to file
   */
  saveKnowledgeBase(outputPath, modelName) {
    const data = {
      modelName: modelName,
      language: this.language,
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      createdAt: new Date().toISOString(),
      chunks: this.chunks,
      stats: {
        totalChunks: this.chunks.length,
        avgChunkLength: this.chunks.reduce((sum, c) => sum + c.text.length, 0) / this.chunks.length
      }
    };

    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2), 'utf-8');
    console.log(`[RAG Builder] Knowledge base saved to: ${outputPath}`);
    console.log(`[RAG Builder] Total chunks: ${data.stats.totalChunks}`);
    console.log(`[RAG Builder] Average chunk length: ${data.stats.avgChunkLength.toFixed(0)} chars`);

    return data.stats;
  }

  /**
   * Read all supported files from a directory (non-recursive)
   */
  static async readKnowledgeData(inputFolder) {
    let combinedText = '';
    let fileCount = 0;

    // Non-recursive: only read files in the specified folder, no subdirectories
    const entries = fs.readdirSync(inputFolder, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(inputFolder, entry.name);

      // Skip subdirectories (non-recursive mode)
      if (entry.isDirectory()) {
        console.log(`[RAG Builder] Skipping subdirectory: ${entry.name} (non-recursive mode)`);
        continue;
      }

      if (entry.isFile()) {
          const ext = path.extname(entry.name).toLowerCase();
          const relativePath = path.relative(inputFolder, fullPath);

          try {
            let content = '';

            if (ext === '.txt') {
              console.log(`[RAG Builder] Reading TXT: ${relativePath}`);
              content = fs.readFileSync(fullPath, 'utf-8');
              fileCount++;
            }
            else if (ext === '.tex') {
              console.log(`[RAG Builder] Reading TEX: ${relativePath}`);
              content = fs.readFileSync(fullPath, 'utf-8');
              // Remove LaTeX commands
              content = content
                .replace(/\\[a-zA-Z]+(\{[^}]*\}|\[[^\]]*\])?/g, '')
                .replace(/[{}]/g, '')
                .replace(/\$.+?\$/g, '')
                .replace(/\$\$.+?\$\$/gs, '');
              fileCount++;
            }
            else if (ext === '.pdf') {
              console.log(`[RAG Builder] Reading PDF: ${relativePath}`);
              const data = await pdfExtract.extract(fullPath, {});
              content = data.pages
                .map(page => page.content.map(item => item.str).join(' '))
                .join('\n');
              fileCount++;
            }
            else if (ext === '.docx') {
              console.log(`[RAG Builder] Reading DOCX: ${relativePath}`);
              const dataBuffer = fs.readFileSync(fullPath);
              const result = await mammoth.extractRawText({ buffer: dataBuffer });
              content = result.value;
              fileCount++;
            }
            else {
              continue;
            }

            if (content && content.trim().length > 0) {
              combinedText += '\n' + content;
            }

          } catch (error) {
            console.error(`[RAG Builder] Error reading ${relativePath}:`, error.message);
          }
        }
      }

    console.log(`[RAG Builder] Total files read: ${fileCount}`);
    return combinedText;
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);

  if (args.length < 3) {
    console.error('Usage: node buildRAG.js <input-folder> <output-name> <language> [chunk-size] [chunk-overlap]');
    console.error('Example: node backend/buildRAG.js knowledge-data my-rag-model ja 500 50');
    process.exit(1);
  }

  const [inputFolder, outputName, language, chunkSizeStr = '500', chunkOverlapStr = '50'] = args;
  const chunkSize = parseInt(chunkSizeStr);
  const chunkOverlap = parseInt(chunkOverlapStr);

  // Resolve paths
  const projectRoot = path.resolve(__dirname, '..');
  const inputPath = path.resolve(projectRoot, inputFolder);
  const outputPath = path.resolve(projectRoot, 'rag-knowledge', `${outputName}.json`);

  console.log('=== RAG Knowledge Base Builder ===');
  console.log(`Input folder: ${inputPath}`);
  console.log(`Output: ${outputPath}`);
  console.log(`Language: ${language}`);
  console.log(`Chunk size: ${chunkSize} characters`);
  console.log(`Chunk overlap: ${chunkOverlap} characters`);
  console.log('');

  // Check if input folder exists
  if (!fs.existsSync(inputPath)) {
    console.error(`Error: Input folder not found: ${inputPath}`);
    process.exit(1);
  }

  // Create output directory
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Read knowledge data
  console.log('[1/4] Reading knowledge data...');
  const knowledgeText = await RAGBuilder.readKnowledgeData(inputPath);

  if (knowledgeText.trim().length === 0) {
    console.error('Error: No text data found in input folder');
    process.exit(1);
  }

  console.log(`Total characters: ${knowledgeText.length}`);
  console.log('');

  // Build RAG knowledge base
  console.log('[2/4] Chunking text...');
  const builder = new RAGBuilder(language, chunkSize, chunkOverlap);

  console.log('[3/4] Creating embeddings (this may take a while)...');
  await builder.processText(knowledgeText);
  console.log('');

  // Save knowledge base
  console.log('[4/4] Saving knowledge base...');
  const stats = builder.saveKnowledgeBase(outputPath, outputName);
  console.log('');

  console.log('=== Build Complete ===');
  console.log(`Knowledge base saved: ${outputPath}`);
  console.log(`File size: ${(fs.statSync(outputPath).size / 1024 / 1024).toFixed(2)} MB`);
}

main().catch(error => {
  console.error('Error:', error);
  process.exit(1);
});
