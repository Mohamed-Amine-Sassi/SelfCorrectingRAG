

from Correction.RAGState import RAGState
from RAGBase.Retrieve import RetrieveDocuments

def RetrieveNode(vectorstore):
    def retrieve(state: RAGState) -> dict:
        docs = RetrieveDocuments(vectorstore, state["original_question"])  
        return {"contexts": [d.page_content for d in docs]}
    return retrieve
