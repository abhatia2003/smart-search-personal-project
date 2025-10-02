# Semantic Search Chrome Extension

A Chrome extension that enables semantic search on web pages, allowing users to find content by meaning rather than exact keyword matches. Based on the semantic search implementation from your Jupyter notebook using embedding-based similarity.

## Features

- 🔍 **Semantic Search**: Find content by meaning, not just keywords
- 📄 **Page Processing**: Automatically extracts and chunks text from web pages
- 🎯 **Smart Highlighting**: Highlights relevant sections with color-coded relevance
- ⚡ **Fast Search**: Efficient similarity matching with instant results
- 🎨 **Clean UI**: Modern popup interface with search results

## How It Works

The extension implements a simplified version of your notebook's semantic search pipeline:

1. **Text Extraction**: Extracts meaningful text content from web pages
2. **Chunking**: Splits content into manageable chunks (similar to your 10-sentence approach)
3. **Embedding Generation**: Creates vector representations of text chunks
4. **Similarity Search**: Uses dot product similarity to find relevant content
5. **Result Highlighting**: Visually highlights matching sections on the page

## Installation

### Manual Installation (Developer Mode)

1. **Enable Developer Mode** in Chrome:
   - Open Chrome and go to `chrome://extensions/`
   - Toggle "Developer mode" in the top right corner

2. **Load the Extension**:
   - Click "Load unpacked"
   - Select the `semantic-search-extension` folder

3. **Verify Installation**:
   - You should see the extension icon in your toolbar
   - The extension will appear in your extensions list

### From Chrome Web Store (Future)
*This extension is not yet published to the Chrome Web Store*

## Usage

1. **Navigate to any webpage** you want to search
2. **Click the extension icon** in your toolbar
3. **Enter your search query** in the popup
   - Example: "machine learning algorithms"
   - Example: "data analysis techniques"
   - Example: "financial planning"
4. **Click "Search"** to find semantically similar content
5. **View results** ranked by relevance
6. **See highlights** on the actual webpage showing matched sections

### Tips for Better Results

- Use conceptual terms rather than exact phrases
- Try broader topics like "artificial intelligence" instead of specific terms
- The extension works better on content-rich pages (articles, documentation, blogs)
- Results are ranked by semantic similarity scores

## Technical Architecture

### Components

- **`content.js`**: Extracts and processes webpage content
- **`background.js`**: Handles embedding generation and semantic search
- **`popup.js`**: Manages the user interface and search interactions
- **`popup.html`**: Search interface design
- **`styles.css`**: Highlighting and UI styles

### Key Differences from Notebook Implementation

- **Simplified Embeddings**: Uses TF-IDF-inspired approach instead of sentence transformers (due to browser limitations)
- **Web-based Processing**: Adapted for real-time webpage analysis
- **Visual Feedback**: Adds highlighting and UI components
- **Memory Management**: Handles multiple tabs and cleanup

## Limitations

- **Model Complexity**: Uses simplified embedding approach (not full sentence transformers)
- **Performance**: May be slower on very large pages
- **Compatibility**: Some pages may block content script injection
- **Accuracy**: Less accurate than the full sentence transformer model from your notebook

## Future Enhancements

### Planned Features
- [ ] Integration with actual sentence transformer models via web APIs
- [ ] Better text preprocessing and chunking algorithms
- [ ] Export search results functionality
- [ ] Custom similarity thresholds
- [ ] Search history and bookmarking

### Advanced Features
- [ ] Support for PDF files and documents
- [ ] Multi-language support
- [ ] Integration with external embedding services
- [ ] Collaborative search and sharing

## Development

### File Structure
```
semantic-search-extension/
├── manifest.json          # Extension configuration
├── popup.html             # Search interface
├── popup.js              # UI logic
├── content.js            # Page content extraction
├── background.js         # Embedding and search logic
├── styles.css           # Highlighting styles
├── icons/               # Extension icons
└── README.md           # This file
```

### Local Development

1. Make changes to the extension files
2. Go to `chrome://extensions/`
3. Click the refresh icon on your extension
4. Test your changes

### Debugging

- **Popup Issues**: Right-click extension icon → "Inspect popup"
- **Content Script**: Open DevTools on any webpage
- **Background Script**: Go to `chrome://extensions/` → Click "service worker"

## Contributing

Feel free to fork this project and submit pull requests for improvements!

### Ideas for Contributions
- Better embedding models integration
- Improved text chunking algorithms
- Enhanced UI/UX design
- Performance optimizations
- Additional search filters

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

Based on the semantic search implementation from the Jupyter notebook using:
- Sentence transformers concept
- Text chunking and embedding techniques
- Dot product similarity matching
- TF-IDF inspired vectorization (simplified for browser environment)

---

**Note**: This extension provides a simplified version of the semantic search capabilities demonstrated in your notebook. For production use with high accuracy requirements, consider integrating with cloud-based embedding services or more sophisticated ML models.