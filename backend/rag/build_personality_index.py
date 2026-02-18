import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from pathlib import Path

# --- 1. الإعدادات والمسارات ---
SCRIPT_DIR = Path(__file__).resolve().parent

# المصادر الأربعة (تأكد أن المجلدات بنفس هذه الأسماء)
INPUT_JSON = SCRIPT_DIR / "personality_examples.json"  # (1) الخاص
KAGGLE_FOLDER = SCRIPT_DIR / "kaggle_data"             # (2) كاغل
SHIFAA_FOLDER = SCRIPT_DIR / "shifaa_data"             # (3) شفاء
MENTALQA_FOLDER = SCRIPT_DIR / "mentalqa_data"         # (4) MentalQA

# ملفات الإخراج
INDEX_FILE_NAME = SCRIPT_DIR / "personality_rag_index.faiss"
METADATA_FILE_NAME = SCRIPT_DIR / "personality_metadata.json"
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

AGENT_LABELS = ["Cooperative", "Defensive", "Avoidant", "Angry", "Fearful", "Confused", "Neutral"]

# --- 2. دالة تحميل بياناتنا الخاصة (JSON) ---
def load_our_data() -> tuple[list[str], list[dict]]:
    chunks, meta = [], []
    if os.path.exists(INPUT_JSON):
        print(f"Loading {INPUT_JSON.name}...")
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            text, label = item.get("text"), item.get("label")
            if text and label in AGENT_LABELS:
                chunks.append(str(text))
                meta.append({"text": str(text), "label": label, "source": "synthetic_v1", "lang": "en"})
        print(f"✅ Loaded {len(chunks)} chunks from our JSON.")
    return chunks, meta

# --- 3. دالة تحميل كاغل (CSV Folder) ---
def load_kaggle_data() -> tuple[list[str], list[dict]]:
    chunks, meta = [], []
    label_map = {'anger': 'Angry', 'fear': 'Fearful', 'sadness': 'Fearful', 'joy': 'Neutral'}
    
    if KAGGLE_FOLDER.exists():
        print(f"Loading Kaggle data from {KAGGLE_FOLDER.name}...")
        for f in KAGGLE_FOLDER.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                # التأكد من وجود الأعمدة
                if 'text' not in df.columns or 'label' not in df.columns: continue
                
                df['mapped'] = df['label'].map(label_map)
                df = df.dropna(subset=['mapped', 'text'])
                for _, row in df.iterrows():
                    chunks.append(str(row['text']))
                    meta.append({"text": str(row['text']), "label": row['mapped'], "source": "kaggle", "lang": "en"})
            except Exception: pass
        print(f"✅ Loaded {len(chunks)} chunks from Kaggle.")
    else:
        print(f"⚠️ Kaggle folder not found at {KAGGLE_FOLDER}")
    return chunks, meta

# --- 4. دالة تحميل شفاء (CSV Folder - Arabic) ---
def load_shifaa_data() -> tuple[list[str], list[dict]]:
    chunks, meta = [], []
    if SHIFAA_FOLDER.exists():
        print(f"Loading Shifaa data from {SHIFAA_FOLDER.name}...")
        for f in SHIFAA_FOLDER.glob("*.csv"):
            try:
                df = pd.read_csv(f)
                # العمود هو 'Question' حسب الصورة
                col = 'Question' if 'Question' in df.columns else df.columns[0]
                df = df.dropna(subset=[col])
                for _, row in df.iterrows():
                    txt = str(row[col])
                    if len(txt) > 10: # تجاهل النصوص القصيرة جداً
                        chunks.append(txt)
                        # نصنفها Neutral لأننا نبيها عشان "اللغة" والكمية
                        meta.append({"text": txt, "label": "Neutral", "source": "shifaa", "lang": "ar"})
            except Exception: pass
        print(f"✅ Loaded {len(chunks)} chunks from Shifaa.")
    else:
        print(f"⚠️ Shifaa folder not found at {SHIFAA_FOLDER}")
    return chunks, meta

# --- 5. دالة تحميل MentalQA (TSV Folder - Arabic High Quality) ---
def load_mentalqa_data() -> tuple[list[str], list[dict]]:
    chunks, meta = [], []
    label_map = {'Emotional-Support': 'Fearful', 'Information': 'Confused', 'Other': 'Neutral'}
    
    if MENTALQA_FOLDER.exists():
        print(f"Loading MentalQA from {MENTALQA_FOLDER.name}...")
        # البحث في كل المجلدات الفرعية (subfolders)
        for f in MENTALQA_FOLDER.glob("**/*.tsv"):
            try:
                df = pd.read_csv(f, sep='\t')
                
                # 1. حالة الملفات المصنفة (مثل subtask3)
                if 'question' in df.columns and 'classification' in df.columns:
                    df['mapped'] = df['classification'].map(label_map)
                    df = df.dropna(subset=['mapped', 'question'])
                    for _, r in df.iterrows():
                        chunks.append(str(r['question']))
                        meta.append({"text": str(r['question']), "label": r['mapped'], "source": "mentalqa", "lang": "ar"})
                
                # 2. حالة ملفات الأسئلة فقط (مثل subtask1) -> نعتبرها Confused/Neutral
                elif 'question' in df.columns:
                     for _, r in df.iterrows():
                        chunks.append(str(r['question']))
                        meta.append({"text": str(r['question']), "label": "Confused", "source": "mentalqa", "lang": "ar"})
            except Exception: pass
        print(f"✅ Loaded {len(chunks)} chunks from MentalQA.")
    else:
        print(f"⚠️ MentalQA folder not found at {MENTALQA_FOLDER}")
    return chunks, meta

# --- 6. التشغيل الرئيسي ---
def main():
    print("--- Building Hybrid Personality Index (AR/EN) ---")
    
    # جمع البيانات
    c1, m1 = load_our_data()
    c2, m2 = load_kaggle_data()
    c3, m3 = load_shifaa_data()
    c4, m4 = load_mentalqa_data()
    
    all_chunks = c1 + c2 + c3 + c4
    all_meta = m1 + m2 + m3 + m4
    
    if not all_chunks:
        print("❌ ERROR: No data found! Check your folders.")
        return
        
    print(f"\n🔥 Total Combined Examples: {len(all_chunks)}")
    print("   (This creates a powerful, bilingual brain for your agent)")

    # التحويل لـ Vectors
    print(f"Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Generating Embeddings (This might take time)...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    
    # بناء الفهرس وحفظه
    print("Building & Saving Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, np.array(range(len(all_chunks))).astype('int64'))
    
    faiss.write_index(index, str(INDEX_FILE_NAME))
    with open(METADATA_FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
        
    print("✅ DONE! Your RAG is ready.")

if __name__ == "__main__":
    main()