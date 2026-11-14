
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import hashlib


def Embedding(Chunkedtexts, pdf_path):
    if isinstance(Chunkedtexts, str):
        raise TypeError("Embedding expects a list of text chunks, not a string.")

    chunks = list(dict.fromkeys(Chunkedtexts))

    if not chunks:
        raise ValueError("No chunks were provided for embedding.")

    collection_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_chunked"

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        collection_name=collection_name,
        embedding_function=embeddings_model
    )

    # Get existing chunks
    existing_data = vectorstore.get(include=["documents"])
    existing_chunks = set(existing_data["documents"])

    # Keep only chunks that don't already exist
    new_chunks = [
        chunk for chunk in chunks
        if chunk not in existing_chunks
    ]

    if new_chunks:
        vectorstore.add_texts(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to {collection_name}.")
    else:
        print(f"All chunks already exist in {collection_name}.")

    return vectorstore
