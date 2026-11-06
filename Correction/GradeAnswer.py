from dotenv import load_dotenv
from RAGState import RAGState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
import json


load_dotenv()
llm = ChatOpenAI(
            model="openrouter/free",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0
        )

class AnswerGrade(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is supported by the chunks.")
    useful: bool = Field(description="True only if the answer directly addresses the question.")
    unsupported_claims: list[str] = Field(description="Claims not supported by the chunks. Empty if none.")

answer_grader = llm.with_structured_output(AnswerGrade)


def GradeAnswer(state: RAGState) -> RAGState:
    """ Grades the answer based on the question and the retrieved chunks. """
    
    chunks = state["contexts"]
    numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(chunks))
    prompt = f"""Question: {state['original_question']}
    Grade the answer based on the question and the retrieved chunks.
        The answer is considered grounded if it is supported by the chunks.
        The answer is considered useful if it addresses the question.
    Chunks: {numbered}
    Question: {state['original_question']}
    Answer: {state['answer']}
    Return a JSON object with two boolean fields: grounded and useful.
    """
    try:
        g = answer_grader.invoke(prompt)
        return {"grounded": g.grounded, "useful": g.useful,
                "feedback": "; ".join(g.unsupported_claims)}
    except Exception:
        return {"grounded": True, "useful": True, "feedback": ""}
    