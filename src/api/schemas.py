from pydantic import BaseModel, Field
from typing import List

class PreprocessRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    remove_stopwords: bool = True
    stem: bool = False
    lemma: bool = False
    min_len: int = 0

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000)

class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)
    
class RetrieveRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
