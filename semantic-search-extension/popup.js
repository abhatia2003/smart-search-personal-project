// Popup script for the Semantic Search Extension
class PopupController {
  constructor() {
    this.searchInput = document.getElementById('searchInput');
    this.searchBtn = document.getElementById('searchBtn');
    this.clearBtn = document.getElementById('clearBtn');
    this.autoProcessToggle = document.getElementById('autoProcessToggle');
    this.status = document.getElementById('status');
    this.loading = document.getElementById('loading');
    this.results = document.getElementById('results');
    
    this.currentTabId = null;
    this.pageProcessed = false;
    this.currentTabUrl = null;
    
    this.initializeEventListeners();
    this.checkCurrentTab();
  }

  initializeEventListeners() {
    this.searchBtn.addEventListener('click', () => this.performSearch());
    this.clearBtn.addEventListener('click', () => this.clearResults());
    this.autoProcessToggle.addEventListener('change', () => this.saveAutoProcessSetting());
    
    // Enable search on Enter key
    this.searchInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.performSearch();
      }
    });

    // Auto-focus search input
    this.searchInput.focus();
  }

  async checkCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      this.currentTabId = tab.id;
      this.currentTabUrl = tab.url;
      
      // Check if we can access this tab
      if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
        this.showStatus('Cannot search on this page', 'error');
        this.searchBtn.disabled = true;
        return;
      }
      // Load per-site auto-process setting
      this.loadAutoProcessSetting();
      
      this.showStatus('Ready to search this page');
    } catch (error) {
      console.error('Error checking current tab:', error);
      this.showStatus('Error accessing current tab', 'error');
    }
  }

  showStatus(message, type = 'info') {
    this.status.textContent = message;
    this.status.className = `status ${type}`;
  }

  showLoading(show = true) {
    this.loading.style.display = show ? 'block' : 'none';
    this.searchBtn.disabled = show;
    this.clearBtn.disabled = show;
  }

  async performSearch() {
    const query = this.searchInput.value.trim();
    if (!query) {
      this.showStatus('Please enter a search query', 'error');
      return;
    }

    try {
      this.showLoading(true);
      this.showStatus('Processing page content...');
      
      // First, extract text chunks from the page
      if (!this.pageProcessed) {
        await this.extractAndProcessText();
      }
      
      // Perform semantic search
      this.showStatus('Searching for similar content...');
      await this.searchContent(query);
      
    } catch (error) {
      console.error('Search error:', error);
      this.showStatus('Search failed. Please try again.', 'error');
    } finally {
      this.showLoading(false);
    }
  }

  async extractAndProcessText() {
    return new Promise((resolve, reject) => {
      // Extract text from content script
      chrome.tabs.sendMessage(this.currentTabId, { action: 'extractText' }, async (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error('Could not communicate with page content'));
          return;
        }

        if (response && response.chunks) {
          // Send chunks to backend /embed endpoint for processing and storage
          try {
            const backendUrl = window.__SEMANTIC_BACKEND_URL__ || 'http://localhost:8000';
            const sourceUrl = this.currentTabUrl || window.location.href;
            const resp = await fetch(`${backendUrl}/embed`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ chunks: response.chunks, source: sourceUrl })
            });
            const json = await resp.json();
            if (resp.ok) {
              this.pageProcessed = true;
              this.showStatus(`Processed ${json.chunks_added} text sections`);
              resolve();
            } else {
              reject(new Error(json.detail || 'Failed to process text chunks'));
            }
          } catch (err) {
            reject(err);
          }
        } else {
          reject(new Error('No text content found on page'));
        }
      });
    });
  }

  // Storage key name for per-site auto-process setting
  autoProcessKeyForUrl(url) {
    try {
      const u = new URL(url);
      return `autoProcess_${u.hostname}`;
    } catch (e) {
      return `autoProcess_unknown`;
    }
  }

  async loadAutoProcessSetting() {
    if (!this.currentTabUrl) return;
    const key = this.autoProcessKeyForUrl(this.currentTabUrl);
    chrome.storage.local.get([key], (items) => {
      // Default: enabled (true) if not set
      const val = items[key];
      if (val === undefined) {
        this.autoProcessToggle.checked = true;
      } else {
        this.autoProcessToggle.checked = Boolean(val);
      }
    });
  }

  saveAutoProcessSetting() {
    if (!this.currentTabUrl) return;
    const key = this.autoProcessKeyForUrl(this.currentTabUrl);
    const val = this.autoProcessToggle.checked;
    const payload = {};
    payload[key] = val;
    chrome.storage.local.set(payload, () => {
      this.showStatus(val ? 'Auto-process enabled for this site' : 'Auto-process disabled for this site');
    });
  }

  async searchContent(query) {
    return new Promise((resolve, reject) => {
      (async () => {
        try {
          const backendUrl = window.__SEMANTIC_BACKEND_URL__ || 'http://localhost:8000';
          const resp = await fetch(`${backendUrl}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 5 })
          });
          const json = await resp.json();
          if (resp.ok && json.results) {
            this.displayResults(json.results, query);
            // send local indices to content script
            const indices = json.results.map(r => r.global_index !== undefined ? r.global_index : null).filter(i => i !== null);
            chrome.tabs.sendMessage(this.currentTabId, {
              action: 'highlightResults',
              indices: indices
            });
            resolve();
          } else {
            reject(new Error(json.detail || 'Search failed'));
          }
        } catch (err) {
          reject(err);
        }
      })();
    });
  }

  displayResults(results, query) {
    if (results.length === 0) {
      this.results.innerHTML = `
        <div class="result-item">
          <div class="result-text">No similar content found for "${query}"</div>
        </div>
      `;
      this.showStatus('🔍 No results found');
      return;
    }

    // Group consecutive results that have the same score (within epsilon)
    const epsilon = 1e-6;
    const groups = [];
    let currentGroup = null;

    results.forEach((r, i) => {
      const score = (r.score !== undefined && r.score !== null) ? Number(r.score) : null;
      const globalIndex = r.global_index ?? r.globalIndex ?? r.globalIndex;
      const text = r.text ?? r.chunk ?? '';

      if (!currentGroup) {
        currentGroup = { score, texts: [text], indices: [globalIndex] };
        return;
      }

      const prevScore = currentGroup.score;
      const equalScore = (prevScore === null && score === null) || (prevScore !== null && score !== null && Math.abs(prevScore - score) <= epsilon);

      if (equalScore) {
        // append to current group
        currentGroup.texts.push(text);
        currentGroup.indices.push(globalIndex);
      } else {
        groups.push(currentGroup);
        currentGroup = { score, texts: [text], indices: [globalIndex] };
      }
    });
    if (currentGroup) groups.push(currentGroup);

    // Render groups
    this.results.innerHTML = '';

    const escapeHtml = (str) => String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');

    groups.forEach((g, gi) => {
      const scoreLabel = (g.score !== null && g.score !== undefined) ? g.score.toFixed(3) : 'N/A';
      const combinedText = g.texts.join(' \n---\n ');
      const displayText = this.truncateText(combinedText, 300);

      const div = document.createElement('div');
      div.className = 'result-item';
      div.setAttribute('data-indices', JSON.stringify(g.indices.filter(i => i !== undefined && i !== null)));

      const scoreDiv = document.createElement('div');
      scoreDiv.className = 'result-score';
      scoreDiv.textContent = `Result ${gi + 1} (Score: ${scoreLabel})`;

      const textDiv = document.createElement('div');
      textDiv.className = 'result-text';
      textDiv.innerHTML = escapeHtml(displayText).replace(/\n---\n/g, '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:8px 0;">');

      div.appendChild(scoreDiv);
      div.appendChild(textDiv);

      // Click to highlight all grouped indices
      div.addEventListener('click', () => {
        const indices = JSON.parse(div.getAttribute('data-indices') || '[]');
        chrome.tabs.sendMessage(this.currentTabId, { action: 'highlightResults', indices });
      });

      this.results.appendChild(div);
    });

    this.showStatus(`Found ${groups.length} similar grouped sections`);
  }

  highlightResults(results) {
    const indices = results.map(result => result.index);
    chrome.tabs.sendMessage(this.currentTabId, {
      action: 'highlightResults',
      indices: indices
    });
  }

  clearResults() {
    this.results.innerHTML = '';
    this.searchInput.value = '';
    this.showStatus('Results cleared');
    
    // Clear highlights on page
    chrome.tabs.sendMessage(this.currentTabId, { action: 'clearHighlights' });
  }

  truncateText(text, maxLength) {
    if (!text) return '';
    const s = String(text);
    if (s.length <= maxLength) return s;
    return s.substring(0, maxLength) + '...';
  }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new PopupController();
});