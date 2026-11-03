
import json
import time
import os
import shutil

from Chunking import Chunking
from Embedding import Embedding
from Retrieve import Retrieve
from Generate import Generate

PDF_PATH = "files/AIBook2.pdf"
SRC = "gemini-code-1793554602243.json"
OUT = "eval_dataset.json"


# ---------------------------------
# Load existing results
# ---------------------------------

def LoadResults():

    if not os.path.exists(OUT):
        return []

    with open(OUT, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)

        if isinstance(data, list):
            return data

    except json.JSONDecodeError:
        print("Warning: JSON file is corrupted.")
        print("Trying to recover completed questions...")

    backup_path = OUT + ".corrupt.bak"
    if os.path.exists(backup_path):
        backup_path = OUT + f".corrupt.{time.time_ns()}.bak"
    shutil.copy2(OUT, backup_path)

    decoder = json.JSONDecoder()
    results = []

    start = content.find("[") + 1

    if start == 0:
        return results

    while start < len(content):

        while start < len(content) and content[start] in " \n\r\t,":
            start += 1

        if start >= len(content) or content[start] == "]":
            break

        try:
            obj, end = decoder.raw_decode(content, start)

            if isinstance(obj, dict) and "id" in obj:
                results.append(obj)

            start = end

        except json.JSONDecodeError:
            print("Reached incomplete JSON record. Stopping recovery.")
            break

    print(f"Recovered {len(results)} completed questions.")

    return results


# ---------------------------------
# Atomic JSON saving
# ---------------------------------

def SaveResults(results):

    TEMP = OUT + ".tmp"

    with open(TEMP, "w", encoding="utf-8") as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(TEMP, OUT)


def BuildEval():
    with open(SRC, "r", encoding="utf-8") as f:
        qa = json.load(f)

    if not isinstance(qa, list):
        raise ValueError("The source questions file must contain a JSON array.")

    results = LoadResults()
    done = {str(r["id"]): r for r in results if isinstance(r, dict) and "id" in r}
    print(f"Already completed: {len(done)} questions.")

    print("\nInitializing RAG pipeline...")
    chunks = Chunking(PDF_PATH)
    vectorstore = Embedding(chunks, PDF_PATH)
    print("RAG pipeline initialized successfully.")

    for item in qa:
        question_id = str(item["id"])
        if question_id in done:
            continue

        try:
            retrieved_docs = Retrieve(vectorstore, item["question"])
            contexts, answer = Generate(retrieved_docs, item["question"])
            result = {
                "id": item["id"],
                "chapter": item["chapter"],
                "user_input": item["question"],
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": item["answer"]
            }

            results.append(result)
            SaveResults(results)
            done[question_id] = result
            print(f"Q{item['id']} done")
            time.sleep(2)

        except Exception as e:
            print(f"Q{item['id']} failed: {e}")
            time.sleep(10)

    print("\nEvaluation dataset generation finished.")
    print(f"Total completed questions: {len(results)}")
    print(f"Output: {OUT}")
    return results


if __name__ == "__main__":
    BuildEval()
