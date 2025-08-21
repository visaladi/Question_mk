from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


Difficulty = Literal["easy","medium","hard"]
Bloom = Literal["Remember","Understand","Apply","Analyze","Evaluate","Create"]


class MCQItem(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    answer_index: int
    rationale: str
    source_pages: List[int] = Field(default_factory=list)
    bloom: Bloom
    difficulty: Difficulty


class MCQBatch(BaseModel):
    items: List[MCQItem]


class EssayItem(BaseModel):
    question: str
    bloom: Bloom
    difficulty: Difficulty
    target_keywords: List[str] = Field(default_factory=list)
    rubric_bullets: List[str] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)


class EssayBatch(BaseModel):
    items: List[EssayItem]


BAD_MODEL_VALUES = {"", " ", "string", "null", "none", "None", "String"}

class GenerateReq(BaseModel):
    topic: str = "full document"
    count: int = 5
    difficulty: Literal["easy","medium","hard"] = "medium"
    qtype: Literal["mcq","essay"] = "mcq"
    llm_backend: Literal["ollama","hf"] = "ollama"
    llm_model: Optional[str] = None

    @field_validator("llm_model")
    def _clean_model(cls, v):
        if v is None:
            return v
        return None if v in BAD_MODEL_VALUES else v