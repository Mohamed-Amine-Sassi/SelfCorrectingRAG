from Retrieve import Retrieve
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def Generate(question, pdf_path):
    docs = Retrieve(question, pdf_path)
    llm = ChatOpenAI(
        model="openrouter/free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    Prompt=f"""
    You are a helpful assistant. Use the following context to answer the question.
    Context: {docs}
    Question: {question}
    Answer:
    """
    response = llm.invoke(Prompt)
    return response.content