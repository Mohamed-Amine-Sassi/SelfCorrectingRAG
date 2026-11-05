from typing import TypedDict
class RAGState(TypedDict):
    question: str          # current question (may be rewritten)
    original_question: str # never changes
    contexts: list[str]    # retrieved chunks
    answer: str
    retries: int           # loop counter
    grounded: bool   # v3: is the answer supported by the contexts?
    useful: bool     # v3: does the answer address the question?