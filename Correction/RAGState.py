from typing import TypedDict

class RAGState(TypedDict):
    question: str            # current search query (may be rewritten)
    original_question: str   # never changes
    contexts: list[str]      # retrieved chunks
    answer: str
    retries: int             # search retries (rewrite_query)
    generations: int         # answer attempts (generate)
    grounded: bool           # v3: answer supported by the contexts?
    useful: bool             # v3: answer addresses the question?
    feedback: str            # v3: unsupported claims, passed back to generate