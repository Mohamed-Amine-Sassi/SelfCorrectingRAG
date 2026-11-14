
import time
from langchain_core.callbacks import BaseCallbackHandler

from RAGBase.Embedding import Embedding
from RAGBase.Chunking import Chunking
from Correction.Graph import BuildGraph


class LLMTimer(BaseCallbackHandler):
    """Logs every LLM call: model, duration, tokens."""

    def __init__(self):
        self.calls = []
        self._starts = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        dur = time.time() - self._starts.pop(run_id, time.time())
        usage = (response.llm_output or {}).get("token_usage", {}) or {}
        model = (response.llm_output or {}).get("model_name", "?")

        self.calls.append((model, dur, usage.get("completion_tokens")))

        print(
            f"  [LLM] {model:40s} {dur:6.1f}s  "
            f"completion_tokens={usage.get('completion_tokens')}"
        )

    def on_llm_error(self, error, *, run_id, **kwargs):
        dur = time.time() - self._starts.pop(run_id, time.time())
        print(f"  [LLM ERROR] after {dur:.1f}s: {type(error).__name__}: {error}")


# --- Stage timing ---
t0 = time.time()
chunks = Chunking("files/AIBook2.pdf")
print(f"Chunking: {time.time()-t0:.1f}s ({len(chunks)} chunks)")

t0 = time.time()
vectorstore = Embedding(chunks, "files/AIBook2.pdf")
print(f"Embedding: {time.time()-t0:.1f}s")

app = BuildGraph(vectorstore)

initial_state = {
    "original_question": "What are the key aspects of self-governance in agentic systems, and how does self-modeling help agents adapt to changing circumstances?",
    "question": "",
    "contexts": [],
    "retries": 0,
    "generations": 0,
}


# --- Run graph and display only final result ---
timer = LLMTimer()
config = {"recursion_limit": 30, "callbacks": [timer]}

t_total = time.time()

try:
    result = app.invoke(initial_state, config=config)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    for key, value in result.items():
        print(f"\n{key}:")
        print(value)

except Exception as e:
    print(f"Graph execution failed: {type(e).__name__}: {e}")


# --- Execution summary ---
print(f"\nTOTAL: {time.time()-t_total:.1f}s | LLM calls: {len(timer.calls)}")

for model in sorted({c[0] for c in timer.calls}):
    ds = [c[1] for c in timer.calls if c[0] == model]
    print(
        f"  {model}: {len(ds)} calls, "
        f"avg {sum(ds)/len(ds):.1f}s, "
        f"max {max(ds):.1f}s"
    )
