from RAGBase.Embedding import Embedding
from RAGBase.Extracting import Extracting
from RAGBase.Chunking import Chunking

from Correction.Graph import BuildGraph

# 1. Load PDF
# 2. Split PDF into chunks
chunks = Chunking("files/AIBook2.pdf")

# 3. Initialize vectorstore
vectorstore = Embedding(chunks, "files/AIBook2.pdf")

# 4. Build graph
app = BuildGraph(vectorstore)

# 5. Initial state
initial_state = {
    "original_question": "what's the voice of the lion named?",
    "question": "",
    "contexts": [],
    "retries": 0,
    "generations": 0
}

# 6. Execute graph
try:
    result = app.invoke(
        initial_state,
        config={"recursion_limit": 30}
    )

    print("\n========== FINAL RESULT ==========")
    print(result)

except Exception as e:
    print(f"Graph execution failed: {type(e).__name__}: {e}")