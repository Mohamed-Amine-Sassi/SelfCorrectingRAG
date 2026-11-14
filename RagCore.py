from RAGBase.Chunking import Chunking
from RAGBase.Embedding import Embedding 
from RAGBase.Retrieve import RetrieveDocuments
from RAGBase.Generate import GenerateAnswer

def build_vectorstore(pdf_path):
    return Embedding(Chunking(pdf_path), pdf_path)

def run_baseline(vectorstore, question):
    docs = RetrieveDocuments(vectorstore, question)
    contexts, answer = GenerateAnswer(docs, question)
    return {"answer": answer, "contexts": contexts}

def run_with_trace(graph, question):
    state = {"question": question, "original_question": question,
             "retries": 0, "generations": 0}
    for step in graph.stream(dict(state), stream_mode="updates"):
        for node, upd in step.items():
            state.update(upd)
            if node == "RetrieveDocuments":
                yield {"node": node, "text": f"Retrieved {len(upd['contexts'])} chunks"}
            elif node == "GradeDocuments":
                yield {"node": node, "text": f"Kept {len(upd['contexts'])} relevant chunks"}
            elif node == "Rewrite":
                yield {"node": node, "text": f"Rewrote the search query: {upd['question']}"}
            elif node == "Generate":
                yield {"node": node, "text": "Generated an answer"}
            elif node == "AnswerGrade":
                yield {"node": node, "text": f"Grounded: {upd['grounded']} | Useful: {upd['useful']}"}
            else:
                yield {"node": node, "text": "No relevant information found"}
    yield {"node": "done", "answer": state.get("answer"), "contexts": state.get("contexts", [])}


