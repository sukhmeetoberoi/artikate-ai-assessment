import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "section_02_rag/vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.json")

class Retriever:
    def __init__(self):
        print("Initializing Local Retriever (all-MiniLM-L6-v2)...")
        # Load local embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.metadata = []

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a piece of text using local sentence-transformers."""
        try:
            text = text.replace("\n", " ")
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def build_index(self, chunks: List[Dict]):
        """Build and save FAISS index from chunks."""
        print("Building FAISS index using local embeddings...")
        embeddings = []
        self.metadata = []

        for chunk in chunks:
            emb = self.get_embedding(chunk["text"])
            if emb:
                embeddings.append(emb)
                self.metadata.append(chunk["metadata"] | {"text": chunk["text"]})

        if not embeddings:
            print("No embeddings generated. Index build failed.")
            return

        embeddings_np = np.array(embeddings).astype('float32')
        # L2 normalization for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_np)

        # Save index and metadata
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
            
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)
        print(f"Index built and saved to {INDEX_PATH}")

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
                self.index = faiss.read_index(INDEX_PATH)
                with open(METADATA_PATH, "r") as f:
                    self.metadata = json.load(f)
                print("Local index and metadata loaded successfully.")
                return True
            else:
                print("Index files not found. You may need to rebuild the index.")
                return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve top-k chunks for a query."""
        if self.index is None:
            if not self.load_index():
                return []

        query_emb = self.get_embedding(query)
        if not query_emb:
            return []

        query_emb_np = np.array([query_emb]).astype('float32')
        faiss.normalize_L2(query_emb_np)

        distances, indices = self.index.search(query_emb_np, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.metadata[idx], float(distances[0][i])))
        
        return results

if __name__ == "__main__":
    retriever = Retriever()
    CHUNKS_FILE = os.path.join(VECTOR_STORE_DIR, "chunks.json")
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r") as f:
            chunks = json.load(f)
        retriever.build_index(chunks)
    else:
        print("chunks.json not found. Run ingest.py first.")
