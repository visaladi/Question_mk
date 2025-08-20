from sentence_transformers import SentenceTransformer
import numpy as np, faiss
from typing import List, Tuple


class EmbedStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Tuple[str, List[int]]] = [] # (chunk_text, source_pages)


    def build(self, chunks: List[Tuple[str, List[int]]]):
        vecs = self.model.encode([c[0] for c in chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.array(vecs, dtype="float32"))
        self.meta = chunks


    def topk(self, query: str, k: int = 4) -> List[Tuple[str, List[int]]]:
        qv = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(qv, dtype="float32"), k)
        out = []
        for idx in I[0]:
        chunk, pages = self.meta[idx]
        out.append((chunk, pages))
        return out