console.log("📄 Content script injected!");

document.addEventListener("keydown", (e) => {
  if (e.key === "/" && e.ctrlKey) {
    alert("Smart Search activated!");
  }
});