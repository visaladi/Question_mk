import json, os, random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample, SentencesDataset, evaluation
from torch.utils.data import DataLoader

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = "models/custom-embed"

def load_pairs(path="data/embed_pairs.json"):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            q = row["query"].strip()
            p = row["positive"].strip()
            negs = row.get("negatives", [])
            pairs.append((q, p, negs))
    return pairs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    pairs = load_pairs()
    random.shuffle(pairs)

    # MultipleNegativesRankingLoss works great with (query, positive) pairs
    train_examples = [InputExample(texts=[q, p]) for q, p, _ in pairs]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Optional: small dev set for sanity
    dev_queries = [q for q, _, _ in pairs[:100]]
    dev_corpus  = [p for _, p, _ in pairs[:100]]
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries={f"q{i}": q for i, q in enumerate(dev_queries)},
        corpus={f"d{i}": d for i, d in enumerate(dev_corpus)},
        relevant_docs={f"q{i}": {f"d{i}"} for i in range(len(dev_queries))}
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,                       # increase if you have more data/GPU
        warmup_steps=100,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=500,
        output_path=OUT_DIR
    )

if __name__ == "__main__":
    main()
