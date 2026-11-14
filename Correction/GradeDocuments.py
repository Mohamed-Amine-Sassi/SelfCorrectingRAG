from dotenv import load_dotenv
from pydantic import BaseModel, Field
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
class Grades(BaseModel):
    """ A structured output class for grading the answer. """
    relevant_ids: list[int] = Field(description="Numbers of the relevant chunks. Empty list if none.")

grader = llm.with_structured_output(Grades,method="json_mode")

def GradeDocuments(state: RAGState) -> RAGState:
    chunks = state["contexts"]
    numbered = "\n\n".join(f"[{i}]\n{c}" for i, c in enumerate(chunks))

    """ Takes the given retries , rate them and either drop them, drope some or keep them. """
    
    
    prompt = f"""
You are an expert relevance evaluator for a Retrieval-Augmented
Generation (RAG) system.

Your task is to identify which retrieved chunks contain information
that can help answer the user's original question.

IMPORTANT:
A chunk does NOT need to answer the entire question to be relevant.
It is sufficient if it contributes useful information to ANY part
of the question.

Grading rules:

1. Evaluate every chunk independently.

2. Consider semantic relevance, not only exact keyword matches.
   Different terminology may describe the same concept.

3. Keep chunks that contain:
   - Direct answers to any part of the question.
   - Relevant definitions or explanations.
   - Supporting facts, principles, or examples.
   - Information that helps explain relationships between concepts.
   - Background information necessary to understand the answer.

4. For multi-part questions, identify the different information
   requirements and evaluate each chunk against ALL of them.

5. Keep partially relevant chunks if they provide meaningful
   information for at least one part of the question.

6. Do NOT reject a chunk simply because:
   - It does not mention the exact terminology used in the question.
   - It does not cover every aspect of the question.
   - It provides supporting information rather than a direct answer.

7. Reject chunks that:
   - Only share keywords without providing useful information.
   - Discuss unrelated concepts.
   - Contain navigation elements, indexes, or reference lists
     without useful explanatory content.
   - Provide examples that have no meaningful connection to
     the question.

8. Do not assume information that is not present in the chunk.
   Judge only what the retrieved text actually contains.

9. When uncertain, keep a chunk if it has a reasonable,
   meaningful connection to answering the question.
   Prefer retaining useful evidence over prematurely discarding it.

Chunks:
{numbered}

Question: {state['original_question']}

Return ONLY valid JSON in this exact format:
{{"relevant_ids": [0, 2, 4]}}

If no chunks are relevant, return:
{{"relevant_ids": []}}

Use only the provided chunk indices.
Do not invent indices or include explanations outside the JSON.
"""

    ids = sorted(set(grader.invoke(prompt).relevant_ids))
    kept = [chunks[i] for i in ids if 0 <= i < len(chunks)]
    return {"contexts": kept}

##TEST
#state = {"original_question": "What is Beta-VAE?", "question": "What is Beta-VAE?",
#         "contexts": ["Beta-VAE adds a hyperparameter ...", "Index: transformer 7, tools 113 ..."],
#         "answer": "", "retries": 0}
#print(GradeDocuments(state)["contexts"])   # should keep only the first chunk