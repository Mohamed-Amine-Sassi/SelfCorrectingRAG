import json, os
from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface sentence-transformers

load_dotenv()

# Load the dataset
rows = json.load(open("./eval_dataset.json", encoding="utf-8"))
rows = rows[10:40]
samples = [
    SingleTurnSample(
        user_input=r["user_input"],
        retrieved_contexts=r["retrieved_contexts"],
        response=r["response"],
        reference=r["reference"],
    )
    for r in rows
]
dataset = EvaluationDataset(samples=samples)

# Judge model: use something stronger than your generator if you can
judge = LangchainLLMWrapper(ChatOpenAI(
    model="openrouter/free",          # via OpenRouter; small cost
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
))
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # free, local
)

result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
    ],
    llm=judge,
    embeddings=embeddings,
)

print(result)

df = result.to_pandas()
df["id"] = [r["id"] for r in rows]
df["chapter"] = [r["chapter"] for r in rows]
df.to_csv("ragas_results.csv", index=False, encoding="utf-8")