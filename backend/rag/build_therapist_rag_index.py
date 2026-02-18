import json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "Therapy Books"

OUT_INDEX = SCRIPT_DIR / "therapist_rag_index.faiss"
OUT_META = SCRIPT_DIR / "therapist_rag_metadata.json"
MANIFEST = SCRIPT_DIR / "therapist_curation_manifest.json"


def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_words(text: str, max_words: int = 350, overlap: int = 60) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break
        start = max(0, end - overlap)

    return chunks


def extract_pages(pdf_path: Path, page_ranges: List[Tuple[int, int]]) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    out: List[Tuple[int, str]] = []

    for a, b in page_ranges:
        for p in range(a, b + 1):
            idx = p - 1  # manifest pages are 1-based
            if 0 <= idx < len(reader.pages):
                txt = reader.pages[idx].extract_text() or ""
                txt = clean_text(txt)
                if txt:
                    out.append((p, txt))

    return out


def main() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing PDF folder: {PDF_DIR}")

    # BOM-safe read (fixes: JSONDecodeError Unexpected UTF-8 BOM)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    model_name = manifest.get("model_name", "paraphrase-multilingual-mpnet-base-v2")

    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for src in manifest["sources"]:
        pdf_file = PDF_DIR / src["file"]
        if not pdf_file.exists():
            raise FileNotFoundError(f"Missing PDF (check exact filename): {pdf_file}")

        page_ranges = [tuple(x) for x in src["include_pages"]]
        pages = extract_pages(pdf_file, page_ranges)

        for page_num, page_text in pages:
            # chunk each page to keep citations simple
            for chunk in chunk_words(page_text):
                docs.append(chunk)
                metas.append(
                    {
                        "source": "pdf",
                        "source_file": src["file"],
                        "topic": src.get("topic", ""),
                        "tier": int(src.get("tier", 1)),
                        "pdf_page": int(page_num),
                        "text": chunk,
                    }
                )

    if not docs:
        raise RuntimeError("No text extracted. Fix page ranges or PDFs.")

    # Embed
    model = SentenceTransformer(model_name)
    emb = model.encode(docs, show_progress_bar=True)
    emb = np.array(emb, dtype="float32")
    dim = emb.shape[1]

    # FAISS index with explicit IDs so metadata order stays stable
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    ids = np.arange(len(docs), dtype="int64")
    index.add_with_ids(emb, ids)

    faiss.write_index(index, str(OUT_INDEX))
    OUT_META.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE")
    print("Wrote:", OUT_INDEX)
    print("Wrote:", OUT_META)
    print("Chunks:", len(docs))


if __name__ == "__main__":
    main()
