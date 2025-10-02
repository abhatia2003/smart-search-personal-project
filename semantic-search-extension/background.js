// Background script (service worker) for embedding generation and semantic search
// Note: Browser extensions have limitations for ML models, so this is a simplified approach

class SemanticSearchEngine {
  constructor() {
    this.pageEmbeddings = new Map(); // Store embeddings per tab
    this.modelReady = false;
    this.initializeModel();
  }

  // Initialize a simple embedding model (using a web-based approach)
  async initializeModel() {
    try {
      // In a real extension, you might use:
      // 1. A lightweight JS ML library like TensorFlow.js
      // 2. A web service API for embeddings
      // 3. Or pre-computed embeddings for common terms
      
      // For now, we'll use a simplified TF-IDF-like approach
      this.modelReady = true;
      console.log('[Semantic Search] Model initialized');
    } catch (error) {
      console.error('[Semantic Search] Model initialization failed:', error);
    }
  }

  // Simple text vectorization (TF-IDF inspired approach)
  // This is a simplified version - in production you'd use sentence transformers
  createEmbedding(text) {
    const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 2);
    const wordFreq = {};
    
    // Count word frequencies
    words.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    // Create a simple vector based on word frequencies
    // This is much simpler than your sentence transformer approach
    const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
    const vector = [];
    
    // Create features for different word categories
    const categories = {
      technical: ['algorithm', 'function', 'method', 'class', 'variable', 'data', 'structure', 'computer', 'software', 'code', 'programming', 'development'],
      business: ['company', 'business', 'market', 'customer', 'revenue', 'profit', 'strategy', 'management', 'finance', 'sales'],
      academic: ['research', 'study', 'analysis', 'theory', 'paper', 'journal', 'university', 'academic', 'scholar', 'publication'],
      general: ['information', 'example', 'important', 'different', 'system', 'process', 'problem', 'solution', 'result']
    };

    // Calculate category scores
    Object.entries(categories).forEach(([category, categoryWords]) => {
      let score = 0;
      categoryWords.forEach(word => {
        if (wordFreq[word]) {
          score += wordFreq[word] / words.length;
        }
      });
      vector.push(score);
    });

    // Add text length features
    vector.push(words.length / 100); // Normalized word count
    vector.push(text.length / 1000); // Normalized character count

    return vector;
  }

  // Calculate dot product similarity (like in your notebook)
  calculateSimilarity(embedding1, embedding2) {
    if (embedding1.length !== embedding2.length) return 0;
    
    let dotProduct = 0;
    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
    }
    return dotProduct;
  }

  // Process text chunks and create embeddings (similar to your notebook approach)
  async processTextChunks(chunks, tabId) {
    const embeddings = chunks.map(chunk => ({
      chunkIndex: chunk.chunkIndex,
      text: chunk.text,
      embedding: this.createEmbedding(chunk.text)
    }));

    this.pageEmbeddings.set(tabId, embeddings);
    console.log(`[Semantic Search] Created embeddings for ${embeddings.length} chunks`);
    return embeddings;
  }

  // Search function (similar to your retrieve_relevant_resources)
  searchSimilarContent(query, tabId, topK = 5) {
    const embeddings = this.pageEmbeddings.get(tabId);
    if (!embeddings) return [];

    const queryEmbedding = this.createEmbedding(query);
    
    // Calculate similarities
    const scores = embeddings.map((item, index) => ({
      score: this.calculateSimilarity(queryEmbedding, item.embedding),
      index: item.chunkIndex,
      text: item.text
    }));

    // Sort by score and return top results
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }
}

// Initialize the search engine
const searchEngine = new SemanticSearchEngine();

// Handle messages from popup and content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const tabId = sender.tab?.id;

  if (request.action === 'processChunks') {
    searchEngine.processTextChunks(request.chunks, tabId)
      .then(embeddings => {
        sendResponse({ success: true, embeddingCount: embeddings.length });
      })
      .catch(error => {
        console.error('[Semantic Search] Error processing chunks:', error);
        sendResponse({ success: false, error: error.message });
      });
    return true; // Keep message channel open for async response
  } 
  
  else if (request.action === 'search') {
    const results = searchEngine.searchSimilarContent(request.query, tabId, request.topK || 5);
    sendResponse({ results });
  }
});

// Clean up embeddings when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
  searchEngine.pageEmbeddings.delete(tabId);
});

console.log('[Semantic Search] Background script loaded');