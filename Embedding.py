from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from Chunking import Chunking

def Embedding(pdf_path):
    Chunkedtexts = Chunking(pdf_path)
    collection_name = os.path.splitext(os.path.basename(pdf_path))[0]
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(
        Chunkedtexts, 
        embeddings_model,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    print(f"Vectorstore created and persisted for {collection_name}.")
    return vectorstore