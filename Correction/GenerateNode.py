from Correction import RAGState
from RAGBase.Generate import GenerateAnswer

def Generate(state: RAGState) -> dict:
    _, answer = GenerateAnswer(state["contexts"], state["original_question"],
                         state.get("feedback", ""))
    return {"answer": answer, "generations": state.get("generations", 0) + 1}