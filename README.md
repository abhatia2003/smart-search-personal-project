# Smart Search — a smarter Ctrl+F for the web

> The first ever *smart* Ctrl+F (or at least a serious attempt).  
> A **Chrome extension** that upgrades plain text find to **semantic search** on any page.

---

## Build roadmap & status

- [x] **Phase 1 — Page text → chunks + working extension UI.** (See [branch/scan-feature](https://github.com/abhatia2003/smart-search-personal-project/tree/branch/scan-feature))   
  _MVP overlay, extract & segment content from webpages/PDFs._
- [x] **Phase 2 — RAG foundations & docs.**  (See [branch-youtube-tutorial](https://github.com/abhatia2003/smart-search-personal-project/tree/branch-youtube-tutorial))
  _Studied the retrieval-augmented generation pipeline; wrote internal notes to guide implementation._
- [x] **Phase 3 — Integrate RAG into the extension.**  
  _Wire retrieval & ranking into Smart mode; add citations, snippets, and result scoring._
- [ ] **Phase 4 - Testing and Bug Fixing**
  _Fix known issues and enhance UI UX for Users to be able to easily optimize their experience._

---

## Why Smart Search?

Classic Ctrl+F only finds **exact** text matches. If you don’t guess the exact word form, hyphenation, or spelling, you’ll miss things.

**Smart Search** augments your find experience with **semantic search**:
- Finds **concepts**, not just strings (e.g., *heart attack* ≈ *myocardial infarction*).
- Tolerates **typos**, **inflections**, and **synonyms**.
- Ranks results by **meaningful relevance**, not just first occurrence.
- Works across **line breaks**, **tables**, and even **PDFs** rendered in the browser.

---

## What it is

A **Google Chrome extension** you install once, then use everywhere.  
Press **Ctrl+E** (or **Cmd+E** on macOS) and get semantic results, highlighted directly on the page—no context switching, no copying into external tools.

> Works on most webpages and embedded PDFs, with OCR for images/screenshots.

---

## Key Features

- **Semantic Find** — Search by meaning, not exact wording.  
- **Synonym & Paraphrase Awareness** — “car” → “vehicle”, “EV”; “cost” → “price”, “expense”.  
- **Fuzzy Matching** — Handles typos (*recieve* → *receive*) and word variants.  
- **Relevance Ranking** — Best matches first; quick-jump keyboard navigation.  
- **PDF Support** — Smarter find inside Chrome’s PDF viewer.  
- **OCR** — Include text inside images/diagrams.  
- **Scopes** — Page-wide, selection-only, or current section/heading.  
- **Smart Snippets** — Preview sentence/paragraph context before jumping.  

---

## Example Use Cases

- **Research papers (PDF)**  
  Query: *time complexity* → finds “**asymptotic runtime**”, “**O(n log n)**”, “**computational cost**”.

- **Policy / Terms of Service**  
  Query: *cancellation* → “**terminate subscription**”, “**refund window**”, “**revocation**”.

- **E-commerce spec sheets**  
  Query: *battery life* → “**playback time**”, “**mAh capacity**”, “**hours on a single charge**”.

- **Lecture notes / transcripts**  
  Query: *heart attack causes* → “**myocardial infarction etiology**”, “**ischemia**”, “**plaque rupture**”.

- **Code docs / diffs**  
  Query: *error handling* → “**exceptions**”, “**retry logic**”, “**graceful degradation**”, “**fallback**”.

---

## Quick Start

1. **Install** from the Chrome Web Store. 
   Or: `chrome://extensions` → **Developer mode** → **Load unpacked** (this repo).  
2. Open any page or PDF. 
3. In Keyboard Shortcuts → Add **Ctrl+E / Cmd+E** as a shortcut to open the extension
4. Press **Ctrl+E / Cmd+E**.  
5. Type a concept (“liver failure”, “time complexity”, “refund policy”).  
6. Use **Enter / Shift+Enter** or **↑ / ↓** to hop between ranked matches.

---

## How It Works (High Level)

1. **Text extraction** — Collect readable text nodes (and PDF text). OCR for images.  
2. **Chunking** — Segment into sentences/paragraphs with anchors so jumps land precisely.  
3. **Embeddings** — Compute vector embeddings on-device (default) to represent meaning.  
4. **Hybrid search** — Combine **semantic similarity** with **keyword signals** for precision and recall.  
5. **Ranking & highlights** — Show top results with snippet previews, highlight all hits and enable keyboard navigation.  
6. **Privacy** — Page text and embeddings stay local unless you explicitly enable a cloud model.

---

## Controls & Shortcuts

- **Next/Previous match:** **Enter / Shift+Enter** or **↓ / ↑**    
- **OCR:** Enable in **Settings** when you need image text  
- **Model:** Currently **On-device** GPU required to run the service.

---

## Limitations

- Heavily scripted apps may virtualize text, making it highly reliant on OCR in those cases. 
- Complex diagrams/handwriting may not be trackable by OCR model.

---

## Installation

### From Chrome Web Store
- *(Link placeholder)* → **Add to Chrome** → Pin the extension for quick access.

### From Source (Developer Mode)
```bash
git clone https://github.com/your-org/smart-search.git
cd smart-search/extension
# In Chrome: chrome://extensions → Enable Developer mode → Load unpacked → select this folder
