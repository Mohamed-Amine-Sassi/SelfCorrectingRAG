from RAGBase.Generate import Generate
from RAGBase.Chunking import Chunking
from RAGBase.Embedding import Embedding
from RAGBase.Retrieve import Retrieve
from Correction.RAGState import RAGState
if __name__ == "__main__":
    State : RAGState={}
    State["question"] = "AI?"
    State["original_question"] = State["question"]
    pdf_path = "files/AIBook2.pdf"
    chunked_text = Chunking(pdf_path)
    vectorstore = Embedding(chunked_text, pdf_path)
    retrieved_docs = Retrieve(vectorstore, State["question"])
    State["contexts"] = retrieved_docs
    contexts, answer = Generate(State["contexts"], State["original_question"])
    State["answer"] = answer
    print(f"Answer: {answer}")


#sth isn't working, I need to check the code in the middle