from typing import List, Tuple
from .schemas import MCQBatch, EssayBatch
from .prompts import MCQ_SYSTEM, MCQ_USER, ESSAY_SYSTEM, ESSAY_USER
from .embed_store import EmbedStore
from .llm_client import LLMClient


class QuestionGenerator:
    def __init__(self, llm: LLMClient, store: EmbedStore, reranker=None):
        self.llm = llm
        self.store = store
        self.reranker = reranker


    def _retrieve(self, topic: str, k: int = 6) -> List[Tuple[str, list]]:
        snips = self.store.topk(topic, k=k)
        if self.reranker is not None:
            snips = self.reranker.rerank(topic, snips, top_k=4)
        else:
            snips = snips[:4]
        return snips


    def _join(self, snips: List[Tuple[str, list]]) -> str:
        return "\n\n".join([f"[pages {','.join(map(str,p))}]\n{t}" for t, p in snips])


    def make_mcqs(self, topic: str, n: int = 5, difficulty: str = "medium") -> MCQBatch:
        ctx = self._join(self._retrieve(topic))
        payload = self.llm.chat_json(MCQ_SYSTEM, MCQ_USER.format(context=ctx, n=n, difficulty=difficulty), temperature=0.2)
        batch = MCQBatch(**payload)
        for it in batch.items:
            if not (0 <= it.answer_index < 4 and len(it.options) == 4):
        raise ValueError("Invalid MCQ structure.")
        return batch


    def make_essays(self, topic: str, n: int = 5) -> EssayBatch:
        ctx = self._join(self._retrieve(topic))
        payload = self.llm.chat_json(ESSAY_SYSTEM, ESSAY_USER.format(context=ctx, n=n), temperature=0.4)
        return EssayBatch(**payload)