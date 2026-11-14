"""
Evaluate.py - score pre-generated RAG answers with RAGAS and write the results to a CSV.

Judge LLM: NVIDIA NIM hosted API (OpenAI-compatible) with ONE pinned model,
so the scores are reproducible from run to run.

Setup
    pip install ragas langchain-openai langchain-huggingface sentence-transformers pandas python-dotenv
    .env    NVIDIA_API_KEY=nvapi-...        (create the key at build.nvidia.com)

Usage
    python Evaluate.py                              # eval_dataset.json -> ragas_results.csv
    python Evaluate.py --limit 5                    # quick smoke test on the first 5 rows
    python Evaluate.py --full --out results/v3.csv  # also context recall + factual correctness
    python Evaluate.py --judge meta/nvidia/openai/gpt-oss-20b

eval_dataset.json = a JSON list; each item has:
    id, chapter, user_input, retrieved_contexts (list of str), response, reference
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Settings
# --------------------------------------------------
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_JUDGE = "qwen2.5:3b"
# Local, free embeddings (only ResponseRelevancy uses them). English text.
# For French/Arabic documents use "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 10  # rows scored per call; the CSV is saved after every batch
RUN_CONFIG = RunConfig(
    timeout=600,
    max_retries=2,
    max_wait=15,
    max_workers=1
)

ROW_KEYS = {"id", "chapter", "user_input", "retrieved_contexts", "response", "reference"}


def resolve(path_str):
    """Relative paths are resolved next to this script."""
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def load_rows(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        missing = ROW_KEYS - r.keys()
        if missing:
            sys.exit(f"Row id={r.get('id')} is missing keys: {sorted(missing)}")
    return rows[:limit] if limit else rows


def build_judge(model):
    chat = ChatOllama(
    model=model,
    temperature=0,
    num_predict=2048,   # was 512: long JSON answers were probably being cut off
    num_ctx=8192,       # Ollama's default context is small and silently trims long prompts
    keep_alive="10m",
)

    try:
        chat.invoke("Reply with the single word OK.")
    except Exception as e:
        sys.exit(
            f"Judge call failed ({type(e).__name__}): {e}\n"
            f"Check that Ollama is running and model '{model}' is installed."
        )

    return LangchainLLMWrapper(chat, run_config=RUN_CONFIG, bypass_n=True)


def score(rows, metrics, judge, embeddings):
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            retrieved_contexts=r["retrieved_contexts"] or [""],
            response=r["response"],
            reference=r["reference"],
        )
        for r in rows
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RUN_CONFIG,
        show_progress=True,
    )
    df = result.to_pandas()
    df["id"] = [r["id"] for r in rows]
    df["chapter"] = [r["chapter"] for r in rows]
    return df


def metric_columns(df):
    skip = {"user_input", "retrieved_contexts", "response", "reference", "id", "chapter"}
    return [c for c in df.columns if c not in skip]


def main():
    ap = argparse.ArgumentParser(description="Score RAG answers with RAGAS (NVIDIA NIM judge).")
    ap.add_argument("--input", default="eval_dataset.json")
    ap.add_argument("--out", default="ragas_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="only score the first N rows")
    ap.add_argument("--full", action="store_true",
                    help="also compute context recall and factual correctness")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="NIM model used as the judge")
    args = ap.parse_args()

    in_path, out_path = resolve(args.input), resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(in_path, args.limit)
    print(f"{len(rows)} rows | judge: {args.judge} | output: {out_path}")

    judge = build_judge(args.judge)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference()]
    if args.full:
        metrics += [LLMContextRecall(), FactualCorrectness()]

    # Score in batches and save after each one, so a crash never loses everything.
    parts = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        print(f"\nScoring rows {start + 1}-{start + len(batch)} of {len(rows)}...")
        parts.append(score(batch, metrics, judge, embeddings))
        pd.concat(parts, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8")

    df = pd.concat(parts, ignore_index=True)
    cols = metric_columns(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # A failed judge call gives NaN. Re-score those rows once.
    bad = df.index[df[cols].isna().any(axis=1)].tolist()
    if bad:
        print(f"\n{len(bad)} row(s) have missing scores. Retrying them once...")
        retry = score([rows[i] for i in bad], metrics, judge, embeddings)
        df.loc[bad, cols] = retry[cols].apply(pd.to_numeric, errors="coerce").values

    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\n=== AVERAGES (missing scores ignored) ===")
    print(df[cols].mean().round(3).to_string())
    print("\n=== MISSING SCORES (NaN) ===")
    print(df[cols].isna().sum().to_string())
    if "faithfulness" in df.columns:
        print("\n=== 3 LOWEST FAITHFULNESS ===")
        print(df.nsmallest(3, "faithfulness")[["id", "chapter", "faithfulness"]].to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()