console.log("🔧 Background service worker running...");

chrome.runtime.onInstalled.addListener(() => {
  console.log("Smart Search extension installed!");
});