from RAGBase.Chunking import Chunking
from RAGBase.Embedding import Embedding
from Correction.RAGState import RAGState
from Correction.Graph import build_graph         

if __name__ == "__main__":
    State : RAGState={}
    State["question"] = "AI?"
    State["original_question"] = State["question"]
    State["retries"] = 0                           
    State["generations"] = 0                      
    pdf_path = "files/AIBook2.pdf"
    chunked_text = Chunking(pdf_path)
    vectorstore = Embedding(chunked_text, pdf_path)

    graph = build_graph(vectorstore)               
    State = graph.invoke(State)                    

    print(f"Answer: {State['answer']}")