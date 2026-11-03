from RAGBase.Generate import Generate
from RAGBase.Chunking import Chunking
from RAGBase.Embedding import Embedding
from RAGBase.Retrieve import Retrieve
if __name__ == "__main__":
    question = "AI?"
    pdf_path = "files/AIBook2.pdf"
    chunked_text = Chunking(pdf_path)
    vectorstore = Embedding(chunked_text, pdf_path)
    retrieved_docs = Retrieve(vectorstore, question)
    contexts, answer = Generate(retrieved_docs, question)
    print(f"Answer: {answer}")
