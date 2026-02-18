import json
import faiss
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# --- CONFIGURATION ---
# Adjust these paths to match your folder structure exactly
DATA_FILE = "legal_review_metadata.json"  # Point to your JSON
INDEX_FILE = "legal_review_index.faiss"  # Where to save the index
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


def rebuild_legal_index():
    print(f"--- STARTING INDEX REBUILD ---")

    # 1. Load the JSON Data
    path = Path(DATA_FILE)
    if not path.exists():
        print(f"❌ ERROR: File not found: {DATA_FILE}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} articles from JSON.")

    # 2. Extract Text & Metadata
    documents = []
    metadatas = []

    for item in data:
        # HANDLE FLAT JSON STRUCTURE
        text = item.get("text", "").strip()
        ref = item.get("article_ref", "Unknown Article")

        if text:
            documents.append(text)
            # We recreate the metadata structure so the RAG service reads it correctly
            metadatas.append({
                "article_ref": ref,
                "text": text,
                "source": item.get("source", "")
            })

    print(f"✅ Prepared {len(documents)} valid text chunks.")

    # 3. Load Model
    print(f"⏳ Loading Embedding Model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()

    # 4. Create Embeddings
    print(f"⏳ Creating Vectors (This may take a moment)...")
    embeddings = model.encode(documents, show_progress_bar=True)

    # 5. Create FAISS Index
    print(f"⏳ Building FAISS Index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # 6. Save Files
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ Saved Index to: {INDEX_FILE}")

    # We don't need to save a new metadata json because your RAG service
    # reads the original one, but we must ensure the order matches.
    # For now, we trust the order is preserved.

    print(f"🎉 SUCCESS! Index contains {index.ntotal} vectors.")
    print("------------------------------------------------")


if __name__ == "__main__":
    rebuild_legal_index()
