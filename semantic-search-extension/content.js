// Content script for extracting and chunking text from web pages
class WebPageTextExtractor {
  constructor() {
    this.chunks = [];
    this.embeddings = null;
    this.isProcessed = false;
  }

  // Text formatter similar to your notebook
  textFormatter(text) {
    return text.replace(/\s+/g, ' ').trim();
  }

  // Extract text from the webpage
  extractPageText() {
    // Get main content, avoiding scripts, styles, and other non-content elements
    const elementsToExclude = ['script', 'style', 'nav', 'header', 'footer', 'aside'];
    const contentElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, article, section, div, span, li');
    
    let pageText = [];
    contentElements.forEach(element => {
      // Skip if parent is excluded element
      if (elementsToExclude.some(tag => element.closest(tag))) {
        return;
      }
      
      const text = element.innerText || element.textContent;
      if (text && text.trim().length > 20) { // Only include substantial text
        pageText.push({
          text: this.textFormatter(text),
          element: element,
          position: this.getElementPosition(element)
        });
      }
    });
    return pageText;
  }

  // Get element position for highlighting later
  getElementPosition(element) {
    const rect = element.getBoundingClientRect();
    return {
      top: rect.top + window.scrollY,
      left: rect.left + window.scrollX,
      width: rect.width,
      height: rect.height
    };
  }

  // Split text into sentences (simplified version of your spaCy approach)
  splitIntoSentences(text) {
    // Simple sentence splitting - can be improved with better NLP
    return text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
  }

  // Create chunks similar to your notebook approach
  createChunks(pageTexts, chunkSize = 10) {
    const chunks = [];
    
    pageTexts.forEach((pageText, pageIndex) => {
      const sentences = this.splitIntoSentences(pageText.text);
      
      // Group sentences into chunks
      for (let i = 0; i < sentences.length; i += chunkSize) {
        const sentenceChunk = sentences.slice(i, i + chunkSize);
        const joinedChunk = sentenceChunk.join('. ').trim();
        
        if (joinedChunk.length > 30) { // Filter short chunks like your notebook
          chunks.push({
            text: joinedChunk,
            element: pageText.element,
            position: pageText.position,
            chunkIndex: chunks.length,
            charCount: joinedChunk.length,
            wordCount: joinedChunk.split(' ').length,
            tokenCount: joinedChunk.length / 4 // Approximate like in your notebook
          });
        }
      }
    });

    return chunks;
  }

  // Process the page and create chunks
  processPage() {
    if (this.isProcessed) return this.chunks;

    console.log('[Semantic Search] Processing page...');
    const pageTexts = this.extractPageText();
    this.chunks = this.createChunks(pageTexts);
    this.isProcessed = true;

    
    console.log(this.chunks);
    console.log(`[Semantic Search] Created ${this.chunks.length} text chunks`);
    return this.chunks;
  }

  // Highlight search results on the page
  highlightResults(resultIndices) {
    // Remove existing highlights
    this.clearHighlights();

    resultIndices.forEach((index, rank) => {
      const chunk = this.chunks[index];
      if (chunk && chunk.element) {
        const highlight = document.createElement('div');
        highlight.className = `semantic-search-highlight rank-${rank}`;
        highlight.style.cssText = `
          position: absolute;
          background-color: rgba(255, 255, 0, ${0.7 - rank * 0.1});
          border: 2px solid #ff6b35;
          border-radius: 4px;
          pointer-events: none;
          z-index: 10000;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        `;
        
        // Position the highlight over the element
        const pos = chunk.position;
        highlight.style.top = `${pos.top}px`;
        highlight.style.left = `${pos.left}px`;
        highlight.style.width = `${pos.width}px`;
        highlight.style.height = `${pos.height}px`;
        
        document.body.appendChild(highlight);

        // Scroll to first result
        if (rank === 0) {
          chunk.element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    });
  }

  // Clear all highlights
  clearHighlights() {
    const highlights = document.querySelectorAll('.semantic-search-highlight');
    highlights.forEach(highlight => highlight.remove());
  }
}

// Initialize the text extractor
const textExtractor = new WebPageTextExtractor();

// Listen for messages from popup/background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'extractText') {
    const chunks = textExtractor.processPage();
    // Return chunks with local_index and position so backend can store metadata
    sendResponse({ chunks: chunks.map(chunk => ({ text: chunk.text, local_index: chunk.chunkIndex, position: chunk.position })) });
  } else if (request.action === 'highlightResults') {
    textExtractor.highlightResults(request.indices);
    sendResponse({ success: true });
  } else if (request.action === 'clearHighlights') {
    textExtractor.clearHighlights();
    sendResponse({ success: true });
  }
});

// Notify that content script is loaded
console.log('[Semantic Search] Content script loaded');