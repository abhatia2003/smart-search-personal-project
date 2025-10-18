// // Background script (service worker) for embedding generation and semantic search
// // Note: Browser extensions have limitations for ML models, so this is a simplified approach

// class SemanticSearchEngine {
//   constructor() {
//     this.pageEmbeddings = new Map(); // Store embeddings per tab
//     this.modelReady = false;
//     this.initializeModel();
//   }

//   // Initialize a simple embedding model (using a web-based approach)
//   async initializeModel() {
//     try {
//       // In a real extension, you might use:
//       // 1. A lightweight JS ML library like TensorFlow.js
//       // 2. A web service API for embeddings
//       // 3. Or pre-computed embeddings for common terms
      
//       // For now, we'll use a simplified TF-IDF-like approach
//       this.modelReady = true;
//       console.log('[Semantic Search] Model initialized');
//     } catch (error) {
//       console.error('[Semantic Search] Model initialization failed:', error);
//     }
//   }

//   // Simple text vectorization (TF-IDF inspired approach)
//   // This is a simplified version - in production you'd use sentence transformers
//   createEmbedding(text) {
//     const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 2);
//     const wordFreq = {};
    
//     // Count word frequencies
//     words.forEach(word => {
//       wordFreq[word] = (wordFreq[word] || 0) + 1;
//     });

//     // Create a simple vector based on word frequencies
//     // This is much simpler than your sentence transformer approach
//     const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'];
//     const vector = [];
    
//     // Create features for different word categories
//     const categories = {
//       technical: ['algorithm', 'function', 'method', 'class', 'variable', 'data', 'structure', 'computer', 'software', 'code', 'programming', 'development'],
//       business: ['company', 'business', 'market', 'customer', 'revenue', 'profit', 'strategy', 'management', 'finance', 'sales'],
//       academic: ['research', 'study', 'analysis', 'theory', 'paper', 'journal', 'university', 'academic', 'scholar', 'publication'],
//       general: ['information', 'example', 'important', 'different', 'system', 'process', 'problem', 'solution', 'result']
//     };

//     // Calculate category scores
//     Object.entries(categories).forEach(([category, categoryWords]) => {
//       let score = 0;
//       categoryWords.forEach(word => {
//         if (wordFreq[word]) {
//           score += wordFreq[word] / words.length;
//         }
//       });
//       vector.push(score);
//     });

//     // Add text length features
//     vector.push(words.length / 100); // Normalized word count
//     vector.push(text.length / 1000); // Normalized character count

//     return vector;
//   }

//   // Calculate dot product similarity (like in your notebook)
//   calculateSimilarity(embedding1, embedding2) {
//     if (embedding1.length !== embedding2.length) return 0;
    
//     let dotProduct = 0;
//     for (let i = 0; i < embedding1.length; i++) {
//       dotProduct += embedding1[i] * embedding2[i];
//     }
//     return dotProduct;
//   }

//   // Process text chunks and create embeddings (similar to your notebook approach)
//   async processTextChunks(chunks, tabId) {
//     const embeddings = chunks.map(chunk => ({
//       chunkIndex: chunk.chunkIndex,
//       text: chunk.text,
//       embedding: this.createEmbedding(chunk.text)
//     }));

//     this.pageEmbeddings.set(tabId, embeddings);
//     console.log(`[Semantic Search] Created embeddings for ${embeddings.length} chunks`);
//     return embeddings;
//   }

//   // Search function (similar to your retrieve_relevant_resources)
//   searchSimilarContent(query, tabId, topK = 5) {
//     const embeddings = this.pageEmbeddings.get(tabId);
//     if (!embeddings) return [];

//     const queryEmbedding = this.createEmbedding(query);
    
//     // Calculate similarities
//     const scores = embeddings.map((item, index) => ({
//       score: this.calculateSimilarity(queryEmbedding, item.embedding),
//       index: item.chunkIndex,
//       text: item.text
//     }));

//     // Sort by score and return top results
//     scores.sort((a, b) => b.score - a.score);
//     return scores.slice(0, topK);
//   }
// }

// // Initialize the search engine
// const searchEngine = new SemanticSearchEngine();

// // Backend URL used by the extension to store page embeddings
// const BACKEND_URL = 'http://localhost:8000';

// // Track which tabs have already been processed for a given URL
// const processedByTab = new Map(); // tabId -> url

// async function sendChunksToBackend(chunks, source) {
//   if (!Array.isArray(chunks) || chunks.length === 0) {
//     console.warn('[Semantic Search] No chunks to send to backend');
//     return null;
//   }

//   // Check backend availability first
//   const avail = await backendAvailable();
//   if (!avail) {
//     console.warn('[Semantic Search] Backend unavailable - skipping remote embed');
//     return null;
//   }

//   try {
//     const resp = await fetch(`${BACKEND_URL}/embed`, {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ chunks, source })
//     });
//     const json = await resp.json();
//     console.log('[Semantic Search] Backend embed response', json);
//     return json;
//   } catch (err) {
//     // Do not throw - log and return null so callers can fallback
//     console.warn('[Semantic Search] Failed to send chunks to backend:', err);
//     return null;
//   }
// }

// async function clearBackend() {
//   // Only attempt to clear if backend is available
//   const avail = await backendAvailable();
//   if (!avail) return;
//   try {
//     await fetch(`${BACKEND_URL}/clear`, { method: 'POST' });
//     console.log('[Semantic Search] Backend cleared');
//   } catch (err) {
//     console.warn('[Semantic Search] Failed to clear backend:', err);
//   }
// }

// // Ping backend /status with short timeout to check availability
// async function backendAvailable(timeoutMs = 500) {
//   try {
//     const controller = new AbortController();
//     const id = setTimeout(() => controller.abort(), timeoutMs);
//     const resp = await fetch(`${BACKEND_URL}/status`, { signal: controller.signal });
//     clearTimeout(id);
//     return resp.ok;
//   } catch (err) {
//     return false;
//   }
// }

// // Process a tab: clear backend then extract & embed the page content.
// async function processTab(tabId, tabUrl) {
//   try {
//     // Avoid reprocessing same URL for same tab
//     const prev = processedByTab.get(tabId);
//     if (prev && prev === tabUrl) {
//       return;
//     }

//     // Respect per-site auto-process setting stored in chrome.storage.local
//     try {
//       const u = new URL(tabUrl);
//       const key = `autoProcess_${u.hostname}`;
//       chrome.storage.local.get([key], async (items) => {
//         const val = items[key];
//         // If explicitly disabled, skip processing
//         if (val === false) {
//           console.log('[Semantic Search] Auto-process disabled for this site:', u.hostname);
//           return;
//         }

//         // Clear backend to ensure only this page's content is stored
//         await clearBackend();

//         // Clear local embeddings for all other tabs to free memory and avoid stale results
//         for (const [otherTabId] of searchEngine.pageEmbeddings) {
//           if (otherTabId !== tabId) {
//             searchEngine.pageEmbeddings.delete(otherTabId);
//           }
//         }

//         // Ask content script to extract chunks
//         chrome.tabs.sendMessage(tabId, { action: 'extractText' }, async (response) => {
//           if (chrome.runtime.lastError) {
//             console.warn('[Semantic Search] Could not extract from tab', tabId, chrome.runtime.lastError.message);
//             return;
//           }

//           if (response && response.chunks && response.chunks.length > 0) {
//             await sendChunksToBackend(response.chunks, tabUrl);
//             processedByTab.set(tabId, tabUrl);
//           } else {
//             console.log('[Semantic Search] No chunks found for tab', tabId);
//           }
//         });
//       });
//       return;
//     } catch (e) {
//       console.warn('[Semantic Search] processTab storage check failed, proceeding with default processing', e);
//     }
//   } catch (err) {
//     console.error('[Semantic Search] processTab error', err);
//   }
// }

// // Handle messages from popup and content scripts (local search engine still available)
// chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
//   const tabId = sender.tab?.id;

//   if (request.action === 'processChunks') {
//     searchEngine.processTextChunks(request.chunks, tabId)
//       .then(embeddings => {
//         sendResponse({ success: true, embeddingCount: embeddings.length });
//       })
//       .catch(error => {
//         console.error('[Semantic Search] Error processing chunks:', error);
//         sendResponse({ success: false, error: error.message });
//       });
//     return true; // Keep message channel open for async response
//   } 
  
//   else if (request.action === 'search') {
//     const results = searchEngine.searchSimilarContent(request.query, tabId, request.topK || 5);
//     sendResponse({ results });
//   }
// });

// // Auto-process pages when they finish loading or are activated
// chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
//   if (changeInfo.status === 'complete' && tab && tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
//     processTab(tabId, tab.url);
//   }
// });

// chrome.tabs.onActivated.addListener(async (activeInfo) => {
//   try {
//     const tab = await chrome.tabs.get(activeInfo.tabId);
//     if (tab && tab.url && !tab.url.startsWith('chrome://') && !tab.url.startsWith('chrome-extension://')) {
//       processTab(tab.id, tab.url);
//     }
//   } catch (err) {
//     console.error('[Semantic Search] onActivated error', err);
//   }
// });

// // When tab is removed or navigated away, clear backend and forget processed state for that tab
// chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
//   processedByTab.delete(tabId);
//   // Remove local embeddings for this tab
//   searchEngine.pageEmbeddings.delete(tabId);
//   // Clear backend so next page has empty DB
//   clearBackend();
// });

// // Also clear when a tab's URL changes (navigation) so DB always reflects current page only
// chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
//   if (changeInfo.url) {
//     processedByTab.delete(tabId);
//     // Remove local embeddings for this tab; navigation means old content no longer valid
//     searchEngine.pageEmbeddings.delete(tabId);
//     clearBackend();
//   }
// });

// console.log('[Semantic Search] Background script loaded (auto-processing enabled)');