"""
build_personality_index.py
==========================
Builds the personality FAISS index from personality_criteria.json.

Previously this script built from raw example datasets (Kaggle, Shifaa,
MentalQA), but the rag_service loads personality_criteria.json as the
metadata source. That mismatch meant vector IDs pointed to the wrong
entries at query time (bug E).

Fix: embed the clinical criteria profiles directly so that:
  - vector[i] = embedding of profile[i]'s definition + keywords
  - metadata[i] = personality_criteria.json profiles[i]

This makes retrieved entries directly useful for the personality agent
which reads meta.get('id'), meta.get('definition'), meta.get('keywords').

Cosine similarity is used via L2-normalized embeddings + IndexFlatIP.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

SCRIPT_DIR    = Path(__file__).resolve().parent
CRITERIA_FILE = SCRIPT_DIR / "personality_criteria.json"
INDEX_FILE    = SCRIPT_DIR / "personality_rag_index.faiss"
MODEL_NAME    = "paraphrase-multilingual-mpnet-base-v2"


def build_text_for_profile(profile: dict) -> str:
    """
    Combine clinical profile fields into a single embeddable string.
    The more information embedded, the better the retrieval precision.
    """
    parts = [profile.get("id", "")]

    definition = profile.get("definition", "")
    if definition:
        parts.append(definition)

    criteria = profile.get("criteria", [])
    if criteria:
        parts.append(". ".join(criteria))

    keywords = profile.get("keywords", [])
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))

    return ". ".join(filter(None, parts))


def main():
    print("--- Building Personality Index from Clinical Criteria ---")

    if not CRITERIA_FILE.exists():
        print(f"ERROR: {CRITERIA_FILE} not found.")
        return

    data = json.loads(CRITERIA_FILE.read_text(encoding="utf-8"))
    profiles = data.get("profiles", [])

    if not profiles:
        print("ERROR: No profiles found in personality_criteria.json.")
        return

    texts = [build_text_for_profile(p) for p in profiles]
    print(f"Profiles to embed: {len(texts)}")
    for i, t in enumerate(texts):
        print(f"  [{i}] {t[:80]}...")

    print(f"\nLoading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # L2-normalize for cosine similarity via IndexFlatIP
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(texts), dtype="int64")
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, str(INDEX_FILE))

    print(f"\nDone.")
    print(f"  Index : {INDEX_FILE}  ({index.ntotal} vectors, dim={dim}, metric=IP/cosine)")
    print(f"  Meta  : {CRITERIA_FILE}  (already loaded by rag_service — no copy needed)")
    print("\nVector IDs now match personality_criteria.json profiles order.")


if __name__ == "__main__":
    main()
