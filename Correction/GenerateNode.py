

from Correction import RAGState
from RAGBase.Generate import Generate

def generate(state: RAGState) -> dict:
    _, answer = Generate(state["contexts"], state["original_question"],
                         state.get("feedback", ""))
    return {"answer": answer, "generations": state.get("generations", 0) + 1}