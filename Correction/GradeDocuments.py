from dotenv import load_dotenv
from pydantic import BaseModel, Field
from Correction.RAGState import RAGState
from langchain_openai import ChatOpenAI
import os

load_dotenv()
llm = ChatOpenAI(
            model="openrouter/free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0
        )
class Grades(BaseModel):
    """ A structured output class for grading the answer. """
    relevant_ids: list[int] = Field(description="Numbers of the relevant chunks. Empty list if none.")

grader = llm.with_structured_output(Grades)

def GradeDocuments(state: RAGState) -> RAGState:
    chunks = state["contexts"]
    numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(chunks))

    """ Takes the given retries , rate them and either drop them, drope some or keep them. """
    
    prompt=f"""Grade these retrieved chunks for the question.
        A chunk is relevant only if it contains information that helps answer the question.
        Index pages, reference lists and loosely related text are NOT relevant.
    Chunks: {numbered}
    Question: {state['original_question']}
    Return the numbers of the relevant chunks.
    """
    ids = sorted(set(grader.invoke(prompt).relevant_ids))
    kept = [chunks[i] for i in ids if 0 <= i < len(chunks)]
    return {"contexts": kept}

##TEST
#state = {"original_question": "What is Beta-VAE?", "question": "What is Beta-VAE?",
#         "contexts": ["Beta-VAE adds a hyperparameter ...", "Index: transformer 7, tools 113 ..."],
#         "answer": "", "retries": 0}
#print(GradeDocuments(state)["contexts"])   # should keep only the first chunk