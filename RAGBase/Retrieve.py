from RAGBase.Embedding import Embedding
def RetrieveDocuments(vectorstore,Question):  
    docs = vectorstore.similarity_search(Question,k=4)
    return docs