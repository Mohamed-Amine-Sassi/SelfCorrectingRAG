

from Correction.RAGState import RAGState
from RAGBase.Retrieve import Retrieve

def make_retrieve(vectorstore):
    def retrieve(state: RAGState) -> dict:
        docs = Retrieve(vectorstore, state["question"])   # current (maybe rewritten) query
        return {"contexts": [d.page_content for d in docs]}
    return retrieve
