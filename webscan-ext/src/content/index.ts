import { scanPage } from "./scan";
import { chunkText } from "./chunk";

console.log("[Smart Search] content script loaded.");

// Run scan + chunk + send to background script
(async () => {
  try {
    const textNodes = scanPage();
    const chunks = chunkText(textNodes);
    
    console.log(`Found ${chunks.length} chunks:`);
    chunks.forEach((chunk, index) => {
      console.log(`Chunk ${index + 1}:`, chunk.text);
    });
    
    // Send chunks to background script for storage
    chrome.runtime.sendMessage({
      type: "SAVE_CHUNKS",
      data: {
        url: window.location.href,
        chunks: chunks
      }
    });
    
    console.log(`[Smart Search] Sent ${chunks.length} chunks for indexing`, window.location.href);
  } catch (err) {
    console.error("[Smart Search] Failed to index page:", err);
  }
})();