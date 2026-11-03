from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def Embedding(Chunkedtexts,pdf_path):
    if isinstance(Chunkedtexts, str):
        raise TypeError("Embedding expects a list of text chunks, not a string.")

    chunks = list(dict.fromkeys(Chunkedtexts))
    if not chunks:
        raise ValueError("No chunks were provided for embedding.")

    collection_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_chunked"
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name=collection_name,
        embedding_function=embeddings_model
    )
    if vectorstore._collection.count() == 0:
        vectorstore.add_texts(chunks)
        print(f"Vectorstore created and persisted for {collection_name}.")

    else:
        print(f"Existing vectorstore loaded: {collection_name}")
    return vectorstore