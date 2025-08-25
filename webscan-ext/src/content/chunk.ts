export interface Chunk {
    id: string;
    text: string;
    url: string;
  }
  
  function sentenceSplit(text: string): string[] {
    return text.split(/(?<=[.!?])\s+/);
  }
  
  export function chunkText(paragraphs: string[], url: string = window.location.href): Chunk[] {
    const sentences = paragraphs.flatMap(sentenceSplit);
    const chunks: Chunk[] = [];
  
    const WINDOW = 150;   // words per chunk
    const OVERLAP = 30;   // word overlap
  
    let buffer: string[] = [];
  
    for (const sentence of sentences) {
      const words = sentence.split(/\s+/);
      buffer.push(...words);
  
      while (buffer.length >= WINDOW) {
        const chunkText = buffer.slice(0, WINDOW).join(" ");
        chunks.push({ id: crypto.randomUUID(), text: chunkText, url });
        buffer = buffer.slice(WINDOW - OVERLAP);
      }
    }
  
    if (buffer.length > 0) {
      chunks.push({ id: crypto.randomUUID(), text: buffer.join(" "), url });
    }
  
    return chunks;
  }