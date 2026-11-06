from RAGBase.Retrieve import Retrieve
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(
    model="PUT_A_FIXED_MODEL_NAME_HERE",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

def Generate(docs,question,feedback=""):
    if isinstance(docs, str):
            contexts = [docs] 
    contexts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    context_text = "\n\n".join(contexts)

    Prompt=f"""
    You are a helpful assistant. Use only the context. Leave out anything not stated in it.
    Context: {context_text}
    Question: {question}
    Answer:
    """
    if feedback:
        prompt += (f"\nYour previous answer contained claims NOT supported by the context: "
                   f"{feedback}\nRewrite the answer and remove them.\n")
    prompt += "Answer:"

    answer = llm.invoke(Prompt)
    return contexts, answer.content