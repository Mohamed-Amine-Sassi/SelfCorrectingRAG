from dotenv import load_dotenv
from RAGState import RAGState
from langchain_openai import ChatOpenAI
import os

load_dotenv()
llm = ChatOpenAI(
            model="openrouter/free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0
        )

def Rewrite(state: RAGState) -> dict:
    """ Rewrite the search query after a failed retrieval or an unhelpful answer. """
    chunks = state["contexts"]
    prompt = f"""A search over a document collection did not find good information for this question.
        Write a better search query: keep the same meaning, but use different keywords,
        synonyms or more specific terms that the document is likely to contain.

        Original question: {state['original_question']}
        Query that failed: {state['question']}

    Return only the new query."""

 
    rewritten_question = llm.invoke(prompt).content.strip()
    return {"question": rewritten_question,"retries": state["retries"] + 1}