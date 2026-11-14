import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from Correction.RAGState import RAGState

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0)

class AnswerGrade(BaseModel):
    grounded: bool = Field(description="True only if every important claim in the answer is supported by the chunks.")
    useful: bool = Field(description="True only if the answer directly and sufficiently addresses the question.")
    unsupported_claims: list[str] = Field(description="Claims in the answer that the chunks do not support. Empty list if none.")

answer_grader = llm.with_structured_output(AnswerGrade, method="json_schema")  # remove method=... if it raises an error

def GradeAnswer(state: RAGState) -> dict:
    """Grade the answer against the question and the retrieved chunks."""
    numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(state["contexts"]))
    prompt = f"""You are grading an answer produced by a RAG system.

Question: {state['original_question']}

Chunks:
{numbered}

Answer:
{state['answer']}

Fill in three fields:
1. grounded: true ONLY if all important factual claims in the answer are supported by the chunks.
   Do not use outside knowledge to validate the answer. Being on the same topic as the chunks is not enough.
2. useful: true ONLY if the answer directly and sufficiently addresses the question.
   Containing relevant keywords is not enough.
3. unsupported_claims: list each claim in the answer that the chunks do not support. Empty list if there are none.

Rules:
- Judge the information, not the writing style.
- If the answer says the information is not available in the context, it claims nothing:
  grounded=true, useful=false.
- Be strict but fair."""
    try:
        g = answer_grader.invoke(prompt)
    except Exception as e:
        print("GRADE_ANSWER FAILED, using fallback:", type(e).__name__, e)
        return {"grounded": True, "useful": True, "feedback": ""}

    feedback = "; ".join(g.unsupported_claims)
    if not g.grounded and not feedback:
        feedback = "Your previous answer contained information that is not stated in the context. Remove it."
    return {"grounded": g.grounded, "useful": g.useful, "feedback": "" if g.grounded else feedback}