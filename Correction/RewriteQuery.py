from dotenv import load_dotenv
from Correction.RAGState import RAGState
from langchain_openai import ChatOpenAI
import os
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

def Rewrite(state: RAGState) -> dict:
    """ Rewrite the search query after a failed retrieval or an unhelpful answer. """
    chunks = state["contexts"]
    prompt = f"""A search over a document collection did not find good information for this question.
        Write a better search query: keep the same meaning, but use different keywords,
        Write ONE natural-language question or sentence for searching the document.
        Keep the same meaning, but use different wording or more specific terms.
        Do NOT use quotes, boolean operators (AND/OR), or lists of alternatives.

        Original question: {state['original_question']}
        Query that failed: {state['question']}

    Return only the new query."""

 
    rewritten_question = llm.invoke(prompt).content.strip()
    return {"question": rewritten_question,"retries": state["retries"] + 1}