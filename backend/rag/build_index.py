import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

# --- Config ---
INDEX_FILE_NAME = "therapy_rag_index.faiss"
METADATA_FILE_NAME = "rag_metadata.json"
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

ENGLISH_CSV = "counsel_chat_cleaned.csv"
ARABIC_CSV = "arabic_empathetic_conversations_cleaned.csv"

def load_and_prepare_data() -> tuple[list[str], list[dict]]:
    """
    Loads the cleaned CSVs and prepares a list of all text chunks
    and a corresponding list of metadata dictionaries.
    """
    all_chunks = []
    all_metadata = []

    # 1. Process English CSV
    if os.path.exists(ENGLISH_CSV):
        print(f"Loading {ENGLISH_CSV}...")
        df_en = pd.read_csv(ENGLISH_CSV)
        df_en = df_en.dropna(subset=['therapist_text'])
        
        for text in df_en['therapist_text']:
            all_chunks.append(str(text))
            all_metadata.append({
                "lang": "en",
                "source": "counsel_chat",
                "text": str(text)
            })
        print(f"Loaded {len(df_en)} English chunks.")
    else:
        print(f"WARNING: {ENGLISH_CSV} not found. Skipping.")

    # 2. Process Arabic CSV
    if os.path.exists(ARABIC_CSV):
        print(f"Loading {ARABIC_CSV}...")
        df_ar = pd.read_csv(ARABIC_CSV)
        df_ar = df_ar.dropna(subset=['therapist_text'])
        
        for text in df_ar['therapist_text']:
            all_chunks.append(str(text))
            all_metadata.append({
                "lang": "ar",
                "source": "arabic_empathetic",
                "text": str(text)
            })
        print(f"Loaded {len(df_ar)} Arabic chunks.")
    else:
        print(f"WARNING: {ARABIC_CSV} not found. Skipping.")
        
    return all_chunks, all_metadata

def main():
    print("--- Starting RAG Index Build ---")
    
    # --- 1. Load and Prepare Data ---
    all_chunks, all_metadata = load_and_prepare_data()
    
    if not all_chunks:
        print("❌ ERROR: No data was loaded. Both CSV files might be missing.")
        return

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # --- 2. Load Embedding Model ---
    print(f"Loading embedding model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ ERROR: Could not load SentenceTransformer model. {e}")
        print("Please install it: pip install sentence-transformers")
        return

    # --- 3. Generate Embeddings ---
    print(f"Embedding {len(all_chunks)} chunks... This may take a while.")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    print(f"Embeddings created with dimension: {dimension}")

    # --- 4. Build and Save FAISS Index ---
    print("Building FAISS index...")
    try:
        index = faiss.IndexFlatL2(dimension)
        # We use IndexIDMap to map the vector's position (0, 1, 2...) to its ID
        index = faiss.IndexIDMap(index) 
        
        # Create an array of sequential IDs
        ids = np.array(range(len(all_chunks))).astype('int64')
        
        # Add vectors and their IDs to the index
        index.add_with_ids(embeddings, ids)
        
        print(f"Saving index to {INDEX_FILE_NAME}...")
        faiss.write_index(index, INDEX_FILE_NAME)
        
    except Exception as e:
        print(f"❌ ERROR: Could not build or save FAISS index. {e}")
        print("Please install it: pip install faiss-cpu")
        return

    # --- 5. Save Metadata ---
    print(f"Saving metadata to {METADATA_FILE_NAME}...")
    try:
        with open(METADATA_FILE_NAME, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ ERROR: Could not save metadata. {e}")
        return

    print("\n--- Index Build Complete ---")
    print("You should now have the following new files:")
    print(f"1. {INDEX_FILE_NAME} (Your vector database)")
    print(f"2. {METADATA_FILE_NAME} (The text for your vectors)")
    print("\nYou can now add these files to your GitHub repository.")

if __name__ == "__main__":
    main()