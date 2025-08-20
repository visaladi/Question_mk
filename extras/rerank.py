from typing import List, Tuple
from sentence_transformers import CrossEncoder


# Fast strong baseline cross-encoder
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
def __init__(self, model_name: str = MODEL):
self.ce = CrossEncoder(model_name)


def rerank(self, query: str, snippets: List[Tuple[str, list]], top_k: int = 4):
pairs = [(query, t) for t, _ in snippets]
scores = self.ce.predict(pairs)
ranked = sorted(zip(snippets, scores), key=lambda x: x[1], reverse=True)
return [snip for (snip, _s) in ranked[:top_k]]