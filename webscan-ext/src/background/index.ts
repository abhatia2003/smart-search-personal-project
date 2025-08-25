import { saveChunks } from "../lib/idb";

console.log("🔧 Background service worker running...");

chrome.runtime.onInstalled.addListener(() => {
  console.log("Smart Search extension installed!");
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "SAVE_CHUNKS") {
    const { url, chunks } = message.data;
    
    saveChunks(url, chunks)
      .then(() => {
        console.log(`[Background] Saved ${chunks.length} chunks for ${url}`);
        sendResponse({ success: true });
      })
      .catch((error) => {
        console.error("[Background] Failed to save chunks:", error);
        sendResponse({ success: false, error: error.message });
      });
    
    return true; // Keep message channel open for async response
  }
});