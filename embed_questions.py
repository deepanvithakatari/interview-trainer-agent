"""
Generate embeddings for all questions in unified_questions.json using
IBM's Granite embedding model, with batching and incremental saving.
"""

import json
import time
import requests
from src.granite_client import GraniteClient

INPUT_FILE = "data/unified_questions.json"
OUTPUT_FILE = "data/questions_with_embeddings.json"
BATCH_SIZE = 50
EMBED_MODEL_ID = "ibm/granite-embedding-278m-multilingual"

def get_embeddings_batch(client, texts):
    token = client._get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model_id": EMBED_MODEL_ID,
        "inputs": texts,
        "project_id": client.project_id,
    }
    endpoint = f"{client.api_url}/ml/v1/text/embeddings?version=2023-05-29"
    response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["results"]]

def main():
    client = GraniteClient()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions. Generating embeddings in batches of {BATCH_SIZE}...")

    results = []
    for i in range(0, len(questions), BATCH_SIZE):
        batch = questions[i:i + BATCH_SIZE]
        texts = [q["question"] for q in batch]

        try:
            embeddings = get_embeddings_batch(client, texts)
            for q, emb in zip(batch, embeddings):
                q["embedding"] = emb
            results.extend(batch)
            print(f"  Batch {i // BATCH_SIZE + 1}/{(len(questions) - 1) // BATCH_SIZE + 1} done "
                  f"({len(results)}/{len(questions)} total)")
        except Exception as e:
            print(f"  Batch starting at {i} FAILED: {e}")
            print("  Saving progress so far and stopping.")
            break

        # Save progress after every batch, so a crash doesn't lose work
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f)

        time.sleep(0.5)  # small pause to be gentle on rate limits

    print(f"\nDone. Saved {len(results)} embedded questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
