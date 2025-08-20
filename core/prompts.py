MCQ_SYSTEM = "You generate high-quality MCQs grounded ONLY in the provided context. Never invent facts. Cite page numbers you used."
MCQ_USER = """Context (snippets with page refs):
{context}


Task:
Produce {n} multiple-choice questions (exactly 4 options, one correct).
Rules:
- Difficulty: {difficulty}
- Include Bloom level, rationale, and source_pages.
- Return JSON:
{{
"items":[{{
"question":"", "options":["","","",""],
"answer_index":0, "rationale":"",
"source_pages":[], "bloom":"", "difficulty":""
}}]
}}
"""


ESSAY_SYSTEM = "You create open-ended questions grounded ONLY in the context."
ESSAY_USER = """Context:
{context}


Task:
Create {n} open-ended questions with Bloom, difficulty, target_keywords, rubric_bullets, source_pages.
Return JSON:
{{
"items":[{{
"question":"", "bloom":"", "difficulty":"",
"target_keywords":[], "rubric_bullets":[], "source_pages":[]
}}]
}}
"""