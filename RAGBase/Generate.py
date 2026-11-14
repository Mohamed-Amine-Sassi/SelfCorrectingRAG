from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq


load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

def GenerateAnswer(docs,question,feedback=""):
    if isinstance(docs, str):
        contexts = [docs]
    else:
        contexts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in docs
        ]
    context_text = "\n\n".join(contexts)

    prompt=f"""
   Answer using ONLY facts explicitly stated in the context.
- Every sentence must be supported by the context. If it isn't stated there, leave it out, even if you know it is true.
- Do not add examples, analogies, definitions or background that are not in the context.
- Keep it short: 2-4 sentences, or at most 5 bullet points.
- If the context does not contain the answer, reply exactly: "The context does not contain this information."
- Do not start with "Based on the context".

Context: {context_text}
Question: {question}
Answer:
    """
    if feedback:
        prompt += (f"\nYour previous answer contained claims NOT supported by the context: "
                   f"{feedback}\nRewrite the answer and remove them.\n")
    prompt += "Answer:"

    answer = llm.invoke(prompt)
    return contexts, answer.content