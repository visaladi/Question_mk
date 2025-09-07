# core/prompts.py

# --- MCQ prompt templates ---
MCQ_SYSTEM = """You are a strict exam question generator. Only use the provided context."""
MCQ_USER = """Context (from lecture notes):
{context}

Task:
Generate {n} high-quality multiple-choice questions based ONLY on the above context.

Requirements:
- Difficulty: {difficulty}
- 4 distinct options, one correct
- Provide question, options, answer_index, rationale, bloom, difficulty, source_pages (ints)

Output JSON:
{
  "items": [
    {
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "answer_index": 0,
      "rationale": "...",
      "source_pages": [..],
      "bloom": "...",
      "difficulty": "..."
    }
  ]
}
Return JSON only."""

# --- Essay prompt templates ---
ESSAY_SYSTEM = """You are a strict exam question generator. Only use the provided context."""
ESSAY_USER = """Context (from lecture notes):
{context}

Task:
Generate {n} open-ended exam questions.

Requirements:
- Difficulty: {difficulty}
- Provide question, bloom, difficulty, target_keywords (3-6), rubric_bullets (3-5), source_pages (ints)

Output JSON:
{
  "items": [
    {
      "question": "...",
      "bloom": "...",
      "difficulty": "...",
      "target_keywords": ["...", "..."],
      "rubric_bullets": ["...", "..."],
      "source_pages": [..]
    }
  ]
}
Return JSON only."""
