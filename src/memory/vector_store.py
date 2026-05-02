"""
src/memory/vector_store.py

Episodic memory for the agent using FAISS + sentence-transformers.

Each Q&A turn is embedded and stored. On new queries, the k most
semantically similar past turns are retrieved and injected into the
agent's context window — a lightweight RAG setup without a database.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~80MB, fast


class VectorMemory:
    """In-memory FAISS store for episodic agent memory."""

    def __init__(self):
        print(f"[Memory] Loading embedding model '{EMBED_MODEL}'...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        dim = 384
        self.index = faiss.IndexFlatL2(dim)
        self.records: list[dict] = []  # [{query, answer, lang}]
        print("[Memory] Ready.")

    def store(self, query: str, answer: str, lang: str = "en") -> None:
        """Embed and store a Q&A turn."""
        text = f"Q: {query} A: {answer}"
        embedding = self.embedder.encode([text], normalize_embeddings=True)
        self.index.add(embedding.astype(np.float32))
        self.records.append({"query": query, "answer": answer, "lang": lang})

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieve the k most semantically similar past turns.

        Returns:
            List of formatted strings "Q: ... A: ..." for injection into the prompt.
            Returns [] if memory is empty.
        """
        if self.index.ntotal == 0:
            return []

        k = min(k, self.index.ntotal)
        embedding = self.embedder.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(embedding.astype(np.float32), k)

        results = []
        for idx in indices[0]:
            if idx >= 0:
                r = self.records[idx]
                results.append(f"Q: {r['query']}\nA: {r['answer']}")
        return results

    def clear(self) -> None:
        """Reset memory (e.g. start a new session)."""
        self.index.reset()
        self.records.clear()

    def __len__(self) -> int:
        return self.index.ntotal
