from typing import List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# You may change to microsoft/deberta-v3-base with a NER head if you have one
MODEL = "dslim/bert-base-NER"


_tok = AutoTokenizer.from_pretrained(MODEL)
_mdl = AutoModelForTokenClassification.from_pretrained(MODEL)
_ner = pipeline("ner", model=_mdl, tokenizer=_tok, aggregation_strategy="simple")


KEEP_LABELS = {"PER","ORG","LOC","MISC"}


def extract_keywords(text: str, top_k: int = 30, min_len: int = 3) -> List[str]:
    ents = _ner(text)
    spans = []
    for e in ents:
        if e.get("entity_group") in KEEP_LABELS:
            word = e.get("word"," ").strip()
            if len(word) >= min_len:
                spans.append(word)
    # Deduplicate, keep order
    seen, out = set(), []
    for s in spans:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out[:top_k]