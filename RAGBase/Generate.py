from RAGBase.Retrieve import Retrieve
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def Generate(docs,question):
    contexts = [
        doc.page_content for doc in docs
    ]

    context_text = "\n\n".join(contexts)

    llm = ChatOpenAI(
        model="openrouter/free",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )
    Prompt=f"""
    You are a helpful assistant. Use only the context. Leave out anything not stated in it.
    Context: {context_text}
    Question: {question}
    Answer:
    """
    answer = llm.invoke(Prompt)
    if isinstance(docs, str):
        contexts = [docs]
    else:
        contexts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    return contexts, answer.content