import Dexie, { type Table } from "dexie";
import type { Chunk } from "../content/chunk";

export interface StoredChunk extends Chunk {
  embedding?: number[]; // optional, filled after embeddings are generated
}

class SmartSearchDB extends Dexie {
  chunks!: Table<StoredChunk, string>;

  constructor() {
    super("SmartSearchDB");
    this.version(1).stores({
      chunks: "id,url", // id = primary key, url for per-page lookups
    });
  }
}

export const db = new SmartSearchDB();

export async function saveChunks(url: string, chunks: StoredChunk[]) {
  await db.chunks.where("url").equals(url).delete();
  await db.chunks.bulkAdd(chunks);
}

export async function getChunks(url: string): Promise<StoredChunk[]> {
  return db.chunks.where("url").equals(url).toArray();
}