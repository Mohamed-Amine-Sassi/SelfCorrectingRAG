from Extracting import Extracting
from langchain_text_splitters import RecursiveCharacterTextSplitter

def Chunking(pdf_path):
    extracted_text = Extracting(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(extracted_text)
    print(f"Number of chunks created: {len(chunks)}")
    return chunks
