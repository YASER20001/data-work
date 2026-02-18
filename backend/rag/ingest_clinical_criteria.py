import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "personality_criteria.json"
OUTPUT_INDEX = SCRIPT_DIR / "personality_rag_index.faiss"
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

def main():
    print("--- Building Clinical Rules Index ---")

    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"❌ ERROR: Missing {INPUT_FILE.name}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    profiles = data.get("profiles", [])
    print(f"✅ Loaded {len(profiles)} profiles.")

    # 2. Prepare Text for Embedding
    documents = []
    ids = []
    for i, p in enumerate(profiles):
        # We embed the DEFINITION and CRITERIA so the AI matches user behavior to the rule.
        text = (
            f"Profile: {p['name']}. "
            f"Definition: {p['definition']} "
            f"Signs: {' '.join(p.get('criteria', []))} "
            f"Keywords: {', '.join(p.get('keywords', []))}"
        )
        documents.append(text)
        ids.append(i)

    # 3. Create Embeddings
    print("⏳ generating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents)

    # 4. Save Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, np.array(ids).astype('int64'))

    faiss.write_index(index, str(OUTPUT_INDEX))
    print(f"✅ SUCCESS: Created {OUTPUT_INDEX.name}")

if __name__ == "__main__":
    main()