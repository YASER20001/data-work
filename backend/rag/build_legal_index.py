import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# --- الإعدادات ---
# تأكد أن هذه الأسماء تطابق ملفاتك الموجودة حالياً
JSON_SOURCE_FILE = "legal_review_metadata.json"  # الملف اللي أنت عدلته بيدك
FAISS_OUTPUT_FILE = "legal_review_index.faiss"  # الملف اللي نبي نحدثه
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'


def main():
    print("--- 🔄 تحديث الفهرس بناءً على التعديلات اليدوية ---")

    # 1. قراءة ملف JSON المعدل
    if not os.path.exists(JSON_SOURCE_FILE):
        print(f"❌ خطأ: الملف {JSON_SOURCE_FILE} غير موجود!")
        return

    print(f"📖 جاري قراءة الملف المعدل: {JSON_SOURCE_FILE}...")
    with open(JSON_SOURCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    print(f"✅ تم تحميل {len(texts)} مادة قانونية (بعد الحذف).")

    # 2. تحميل موديل الذكاء الاصطناعي
    print(f"📥 جاري تحميل الموديل: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {e}")
        return

    # 3. تحويل النصوص إلى أرقام (Embeddings)
    print("⚙️  جاري إنشاء الفهرس الجديد (FAISS)...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # 4. بناء وحفظ ملف FAISS الجديد
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(index)

    # نربط الأرقام الجديدة (من 0 إلى عدد المواد الجديد)
    ids = np.array(range(len(texts))).astype('int64')
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, FAISS_OUTPUT_FILE)

    print("-" * 30)
    print(f"🎉 تم بنجاح!")
    print(f"📁 تم تحديث ملف: {FAISS_OUTPUT_FILE}")
    print(f"📊 عدد المواد في الفهرس الآن: {len(texts)}")
    print("الآن الـ Agent والبيانات متطابقين 100%.")


if __name__ == "__main__":
    main()
