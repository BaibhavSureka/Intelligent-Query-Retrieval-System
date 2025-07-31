import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, embeddings: list, chunks: list):
        arr = np.array(embeddings).astype("float32")
        self.index.add(arr)
        self.chunks.extend(chunks)

    def search(self, query_emb: list, top_k: int = 5):
        arr = np.array([query_emb]).astype("float32")
        D, I = self.index.search(arr, top_k)
        results = []
        for idx, i in enumerate(I[0]):
            if 0 <= i < len(self.chunks):
                results.append((self.chunks[i], float(D[0][idx])))
        return results