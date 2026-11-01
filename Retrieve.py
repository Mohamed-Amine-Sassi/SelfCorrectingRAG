from Embedding import Embedding
def Retrieve(Question, pdf_path):
    vectorstore = Embedding(pdf_path)
    docs = vectorstore.similarity_search(Question,k=4)
    return docs