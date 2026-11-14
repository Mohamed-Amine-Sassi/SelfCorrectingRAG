"""
Evaluate.py - score pre-generated RAG answers with RAGAS, judged by a LOCAL Ollama model.

Default judge: qwen2.5:7b.
Results go to a NEW file named after the judge (ragas_results_qwen2.5-7b.csv),
so earlier results (for example the 3B run) are never overwritten.
The CSV is saved after every small batch, and if you start the script again
it RESUMES where it stopped (use --fresh to start over).

Setup
    ollama pull qwen2.5:7b
    pip install ragas langchain-ollama langchain-huggingface sentence-transformers pandas python-dotenv

Usage
    python Evaluate.py                          # all rows, 3 metrics, judge qwen2.5:7b
    python Evaluate.py --metrics faithfulness   # faster: faithfulness only
    python Evaluate.py --sample 30              # 30 rows spread over the file
    python Evaluate.py --limit 10               # first 10 rows only
    python Evaluate.py --fresh                  # ignore an existing output file, start over
    python Evaluate.py --judge qwen2.5:3b       # other model -> its own output file

Overnight (stops the computer from sleeping while it runs):
    systemd-inhibit --what=sleep:idle python Evaluate.py

eval_dataset.json = a JSON list; each item has:
    id, chapter, user_input, retrieved_contexts (list of str), response, reference
--metrics choices: faithfulness, relevancy, precision, recall, correctness
"""
import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Settings
# --------------------------------------------------
DEFAULT_JUDGE = "qwen2.5:7b"
# Local, free embeddings (only ResponseRelevancy uses them). English text.
# For French/Arabic documents use "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 5  # rows per batch; the CSV is saved after every batch
# Slow local model: one call at a time, long timeout.
RUN_CONFIG = RunConfig(timeout=900, max_retries=3, max_wait=30, max_workers=1)

ROW_KEYS = {"id", "chapter", "user_input", "retrieved_contexts", "response", "reference"}
META_COLS = {"user_input", "retrieved_contexts", "response", "reference", "id", "chapter"}

# name -> (RAGAS metric class, rough number of judge calls per row; n = retrieved chunks)
METRICS = {
    "faithfulness": (Faithfulness, lambda n: 2),
    "relevancy": (ResponseRelevancy, lambda n: 3),
    "precision": (LLMContextPrecisionWithReference, lambda n: max(n, 1)),
    "recall": (LLMContextRecall, lambda n: 1),
    "correctness": (FactualCorrectness, lambda n: 4),
}


def resolve(path_str):
    """Relative paths are resolved next to this script."""
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def default_out(judge):
    """qwen2.5:7b -> ragas_results_qwen2.5-7b.csv"""
    safe = re.sub(r"[^A-Za-z0-9.]+", "-", judge).strip("-")
    return f"ragas_results_{safe}.csv"


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        missing = ROW_KEYS - r.keys()
        if missing:
            sys.exit(f"Row id={r.get('id')} is missing keys: {sorted(missing)}")
    return rows


def pick_rows(rows, limit, sample):
    """--sample N keeps N rows spread evenly over the file (all chapters, same rows every run)."""
    if sample and sample < len(rows):
        step = (len(rows) - 1) / (sample - 1) if sample > 1 else 0
        return [rows[i] for i in sorted({round(k * step) for k in range(sample)})]
    return rows[:limit] if limit else rows


def build_judge(model):
    chat = ChatOllama(
        model=model,
        temperature=0,
        num_predict=2048,   # long JSON verdicts must not be cut off
        num_ctx=8192,       # Ollama's default context is small and silently trims long prompts
        keep_alive="30m",
    )
    # Fail fast with a clear message instead of a night of empty scores.
    try:
        chat.invoke("Reply with the single word OK.")
    except Exception as e:
        sys.exit(
            f"Judge call failed ({type(e).__name__}): {e}\n"
            f"Check that Ollama is running and that the model is installed: ollama pull {model}"
        )
    # bypass_n: send n separate requests instead of one request asking for n completions.
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
    return [c for c in df.columns if c not in META_COLS]


def main():
    ap = argparse.ArgumentParser(description="Score RAG answers with RAGAS (local Ollama judge).")
    ap.add_argument("--input", default="eval_dataset.json")
    ap.add_argument("--out", default=None, help="default: ragas_results_<judge>.csv")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="Ollama model used as the judge")
    ap.add_argument("--limit", type=int, default=0, help="only score the first N rows")
    ap.add_argument("--sample", type=int, default=0,
                    help="score N rows spread evenly over the file (same rows every run)")
    ap.add_argument("--metrics", default="faithfulness,relevancy,precision",
                    help="comma list from: " + ", ".join(METRICS))
    ap.add_argument("--full", action="store_true", help="shortcut: all five metrics")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing output file and start over")
    args = ap.parse_args()

    in_path = resolve(args.input)
    out_path = resolve(args.out or default_out(args.judge))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = pick_rows(load_rows(in_path), args.limit, args.sample)
    names = list(METRICS) if args.full else [n.strip() for n in args.metrics.split(",") if n.strip()]
    unknown = [n for n in names if n not in METRICS]
    if unknown:
        sys.exit(f"Unknown metric(s) {unknown}. Choose from: {', '.join(METRICS)}")

    calls = sum(METRICS[n][1](len(r["retrieved_contexts"])) for r in rows for n in names)
    print(f"{len(rows)} rows (ids {rows[0]['id']}..{rows[-1]['id']}) | metrics: {', '.join(names)}")
    print(f"judge: {args.judge} | output: {out_path}")
    print(f"Estimated judge calls: ~{calls}")

    # Resume: keep rows that are already in the output file.
    existing, todo = None, rows
    if out_path.exists() and not args.fresh:
        existing = pd.read_csv(out_path)
        done = set(existing["id"].tolist())
        todo = [r for r in rows if r["id"] not in done]
        print(f"Resuming: {len(done)} rows already in {out_path.name}, {len(todo)} left. "
              "(If you changed --metrics, use --fresh.)")

    judge = build_judge(args.judge)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    metrics = [METRICS[n][0]() for n in names]

    # Score in small batches and save after each one, so nothing is lost if it stops.
    parts = [existing] if existing is not None else []
    t0 = time.time()
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        print(f"\nScoring rows {start + 1}-{start + len(batch)} of {len(todo)}...")
        parts.append(score(batch, metrics, judge, embeddings))
        pd.concat(parts, ignore_index=True).to_csv(out_path, index=False, encoding="utf-8")
        n_done = start + len(batch)
        elapsed = time.time() - t0
        eta = elapsed / n_done * (len(todo) - n_done)
        print(f"Saved. {n_done}/{len(todo)} done | elapsed {elapsed / 60:.0f} min | ETA {eta / 60:.0f} min")

    if not parts:
        sys.exit("Nothing to score.")
    df = pd.concat(parts, ignore_index=True)
    cols = metric_columns(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    # A failed judge call gives NaN. Re-score those rows once (matched by id).
    by_id = {r["id"]: r for r in rows}
    mask = df[cols].isna().any(axis=1) & df["id"].isin(by_id)
    if mask.any():
        ids = df.loc[mask, "id"].tolist()
        print(f"\n{len(ids)} row(s) have missing scores. Retrying them once...")
        retry = score([by_id[i] for i in ids], metrics, judge, embeddings)
        df.loc[mask, cols] = retry[cols].apply(pd.to_numeric, errors="coerce").values

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n=== AVERAGES over {len(df)} rows (missing scores ignored) ===")
    print(df[cols].mean().round(3).to_string())
    print("\n=== MISSING SCORES (NaN) ===")
    print(df[cols].isna().sum().to_string())
    if "faithfulness" in df.columns:
        print("\n=== 3 LOWEST FAITHFULNESS ===")
        print(df.nsmallest(3, "faithfulness")[["id", "chapter", "faithfulness"]].to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""=== AVERAGES over 50 rows (missing scores ignored) ===
faithfulness                            0.952
answer_relevancy                        0.894
llm_context_precision_with_reference    0.878

=== MISSING SCORES (NaN) ===
faithfulness                            1
answer_relevancy                        0
llm_context_precision_with_reference    0

=== 3 LOWEST FAITHFULNESS ===
 id   chapter  faithfulness
 45 Chapter 5      0.000000
  5 Chapter 1      0.625000
 38 Chapter 4      0.666667
 """