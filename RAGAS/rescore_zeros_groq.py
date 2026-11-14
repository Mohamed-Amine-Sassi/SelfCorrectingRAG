"""
rescore_zeros_groq.py - re-score ONLY the rows that got faithfulness = 0, with a stronger judge on Groq.

Why: a small local judge sometimes gives 0 to answers that ARE supported by the chunks.
This script re-scores just those rows with a Groq model and shows old vs new side by side.
It reads everything it needs from your results CSV, so no other file is required.

Setup
    pip install ragas langchain-groq pandas python-dotenv
    .env    GROQ_API_KEY=gsk_...

Usage (the --csv path is relative to this script's folder)
    python RAGAS/rescore_zeros_groq.py                   # every row with faithfulness == 0
    python RAGAS/rescore_zeros_groq.py --n 2             # quick test on the first 2 rows
    python RAGAS/rescore_zeros_groq.py --ids 6,15,22     # chosen ids
    python RAGAS/rescore_zeros_groq.py --judge openai/gpt-oss-20b   # if the 120b quota is low
    python RAGAS/rescore_zeros_groq.py --csv ragas_results.csv
"""
import argparse
import ast
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JUDGE = "openai/gpt-oss-120b"
# One call at a time: Groq's free tier limits tokens per minute, so parallel calls hit 429 errors.
RUN_CONFIG = RunConfig(timeout=180, max_retries=6, max_wait=60, max_workers=1)


def main():
    ap = argparse.ArgumentParser(description="Re-score faithfulness=0 rows with a Groq judge.")
    ap.add_argument("--csv", default="ragas_results.csv", help="results CSV from Evaluate.py")
    ap.add_argument("--ids", default="", help="comma list of ids; default = all rows with faithfulness == 0")
    ap.add_argument("--n", type=int, default=0, help="only re-score the first N selected rows")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="Groq model name")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    df = pd.read_csv(csv_path)
    df["faithfulness"] = pd.to_numeric(df["faithfulness"], errors="coerce")

    if args.ids:
        wanted = [int(x) for x in args.ids.split(",") if x.strip()]
        rows = df[df["id"].isin(wanted)]
    else:
        rows = df[df["faithfulness"] == 0]
    if args.n:
        rows = rows.head(args.n)
    if rows.empty:
        sys.exit("No rows to re-score.")

    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY is missing. Add it to your .env file: GROQ_API_KEY=gsk_...")

    print(f"{len(rows)} rows to re-score: ids {rows['id'].tolist()}")
    print(f"judge: {args.judge} | about {2 * len(rows)} judge calls (faithfulness needs ~2 per row)")

    chat = ChatGroq(model=args.judge, api_key=key, temperature=0, max_retries=3)
    try:  # fail fast with a clear message
        chat.invoke("Reply with the single word OK.")
    except Exception as e:
        sys.exit(f"Judge call failed ({type(e).__name__}): {e}\n"
                 f"Check GROQ_API_KEY and that the model '{args.judge}' is still listed in the Groq console.")
    judge = LangchainLLMWrapper(chat, run_config=RUN_CONFIG, bypass_n=True)

    samples = [
        SingleTurnSample(
            user_input=r.user_input,
            retrieved_contexts=ast.literal_eval(r.retrieved_contexts) or [""],
            response=r.response,
            reference=r.reference,
        )
        for r in rows.itertuples()
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness()],
        llm=judge,
        run_config=RUN_CONFIG,
        show_progress=True,
    )
    new = pd.to_numeric(result.to_pandas()["faithfulness"], errors="coerce").values

    res = pd.DataFrame({
        "id": rows["id"].values,
        "chapter": rows["chapter"].values,
        "old_faithfulness": rows["faithfulness"].values,
        "new_faithfulness": new,
    })
    res.to_csv(BASE_DIR / "rescore_zeros_groq.csv", index=False)

    scored = res["new_faithfulness"].notna()
    print("\n=== OLD (local judge) vs NEW (Groq) ===")
    print(res.round(2).to_string(index=False))
    print(f"\nscored: {int(scored.sum())}/{len(res)} (the rest failed: check the Groq error messages above)")
    print(f"rows now >= 0.7: {int((res['new_faithfulness'] >= 0.7).sum())}")
    print(f"mean of these rows: {res['old_faithfulness'].mean():.3f} -> {res['new_faithfulness'].mean():.3f}")

    # Overall effect if the new values replace the old zeros (rows that failed keep the old value).
    merged = df.set_index("id")["faithfulness"].copy()
    merged.loc[res.loc[scored, "id"].values] = res.loc[scored, "new_faithfulness"].values
    print(f"whole file faithfulness: {df['faithfulness'].mean():.3f} -> {merged.mean():.3f} "
          "(if these re-scored values replace the zeros)")
    print(f"\nSaved to {BASE_DIR / 'rescore_zeros_groq.csv'}")


if __name__ == "__main__":
    main()