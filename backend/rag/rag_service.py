import faiss
import json
import numpy as np
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path

# --- Constants ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

THERAPY_INDEX_FILE    = SCRIPT_DIR / "therapy_rag_index.faiss"
RAG_METADATA_FILE     = SCRIPT_DIR / "rag_metadata.json"

PERSONALITY_INDEX_FILE    = SCRIPT_DIR / "personality_rag_index.faiss"
PERSONALITY_METADATA_FILE = SCRIPT_DIR / "personality_criteria.json"

LEGAL_RAG_INDEX_FILE    = SCRIPT_DIR / "legal_review_index.faiss"
LEGAL_RAG_METADATA_FILE = SCRIPT_DIR / "legal_review_metadata.json"

THERAPIST_INDEX_FILE    = SCRIPT_DIR / "therapist_rag_index.faiss"
THERAPIST_METADATA_FILE = SCRIPT_DIR / "therapist_rag_metadata.json"

# --- Relevance thresholds ---
# IndexFlatIP with L2-normalized embeddings = cosine similarity (range -1 to 1)
# IndexFlatL2 with L2-normalized embeddings = 2 - 2*cosine (range 0 to 2)
_IP_THRESHOLD = 0.25   # cosine > 0.25  (keep moderately relevant results)
_L2_THRESHOLD = 1.50   # distance < 1.5  (equiv to cosine > 0.25 for unit vectors)

# --- Source-file to semantic tag mapping (for LLM prompt context) ---
_SOURCE_TAG_MAP = {
    "CCSA-Motivational-Interviewing":  "[MI Technique]",
    "understanding_mi":                "[MI Technique]",
    "Trauma-Informed":                 "[Trauma Care]",
    "SAMHSA":                          "[Trauma Care]",
    "WHO":                             "[WHO Guideline]",
    "HEC":                             "[Clinical Guide]",
    "Bookshelf":                       "[Clinical Evidence]",
    "counsel_chat":                    "[Therapist Example]",
    "arabic_empathetic":               "[Arabic Example]",
    "synthetic_v1":                    "[Synthetic]",
}


def map_source_tag(meta: Dict[str, Any]) -> str:
    """Map raw metadata to a human-readable semantic label for LLM prompts."""
    src = meta.get("source_file") or meta.get("source") or meta.get("tag") or "example"
    for key, tag in _SOURCE_TAG_MAP.items():
        if key.lower() in src.lower():
            return tag
    if meta.get("topic") == "mi":
        return "[MI Technique]"
    return f"[{str(src)[:20]}]"


def _is_ip_index(index: faiss.Index) -> bool:
    """Return True if the FAISS index uses inner-product (cosine) metric."""
    if isinstance(index, faiss.IndexIDMap):
        return isinstance(index.index, faiss.IndexFlatIP)
    return isinstance(index, faiss.IndexFlatIP)


class RagPipeline:
    """
    Optimized RAG Service with:
      - L2-normalized embeddings for cosine similarity (A)
      - Per-turn embedding cache — single encode per unique text (H + I)
      - Relevance score threshold filtering (B)
      - IndexFlatIP / IndexFlatL2 auto-detection with correct thresholds (E)
      - Parallel multi-index search via ThreadPoolExecutor (Time opt)
      - search_combined() — unified fan-out search across indexes (L)
      - dynamic_k()        — adapts retrieval depth to message complexity (M)
      - deduplicate_snippets() — deduplicates + budgets total snippet length (J)
    """

    def __init__(self, model_name: str = None):
        print("[RagPipeline] Initializing...")

        if not model_name:
            print(f"[RagPipeline] No model name provided. Defaulting to: {DEFAULT_MODEL_NAME}")
            model_name = DEFAULT_MODEL_NAME

        # 1. Load embedding model
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            _ = self.model.encode(["test integrity check"])
            print(f"[RagPipeline] SUCCESS: Loaded embedding model: {model_name} (Dim: {self.dimension})")
        except Exception as e:
            print(f"\n[RagPipeline] CRITICAL ERROR: Could not load SentenceTransformer model!\nReason: {e}")
            self.model = None
            self.dimension = 0

        # 2. Per-turn embedding cache: {md5_hash: np.ndarray}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        # 3. Load all indexes and metadata
        self.therapy_index    = self._load_index(THERAPY_INDEX_FILE,    "TherapyStyle")
        self.therapy_metadata = self._load_metadata(RAG_METADATA_FILE,  "TherapyStyle")

        self.personality_index    = self._load_index(PERSONALITY_INDEX_FILE,    "Personality")
        self.personality_metadata = self._load_metadata(PERSONALITY_METADATA_FILE, "Personality")

        self.legal_review_index    = self._load_index(LEGAL_RAG_INDEX_FILE,    "LegalReview")
        self.legal_review_metadata = self._load_metadata(LEGAL_RAG_METADATA_FILE, "LegalReview")

        self.therapist_index    = self._load_index(THERAPIST_INDEX_FILE,    "Therapist")
        self.therapist_metadata = self._load_metadata(THERAPIST_METADATA_FILE, "Therapist")

        print("[RagPipeline] Initialization complete.")

    # =========================================================================
    # INDEX / METADATA LOADING
    # =========================================================================

    def _load_index(self, file_path: Path, name: str) -> Optional[faiss.Index]:
        if not file_path.exists():
            print(f"[RagPipeline] WARNING: {name} index file not found at {file_path}")
            return None
        try:
            index = faiss.read_index(str(file_path))
            if self.model and index.d != self.dimension:
                print(f"[RagPipeline] WARNING: {name} index dim ({index.d}) != model dim ({self.dimension})")
            metric = "IP (cosine)" if _is_ip_index(index) else "L2"
            print(f"[RagPipeline] Loaded {name} index ({index.ntotal} vectors, metric={metric}).")
            return index
        except Exception as e:
            print(f"[RagPipeline] ERROR: Could not load {name} index from {file_path}: {e}")
            return None

    def _load_metadata(self, file_path: Path, name: str) -> List[Dict[str, Any]]:
        if not file_path.exists():
            print(f"[RagPipeline] WARNING: {name} metadata file not found at {file_path}")
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and "profiles" in data:
                print(f"[RagPipeline] Loaded {name} from 'profiles' key ({len(data['profiles'])} items).")
                return data["profiles"]
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"[RagPipeline] ERROR: Could not load {name} metadata: {e}")
            return []

    # =========================================================================
    # EMBEDDING WITH CACHE (H + I)
    # =========================================================================

    def _embed(self, text: str) -> np.ndarray:
        """
        Encode text to a L2-normalized unit vector with per-turn caching.

        Normalizing enables cosine similarity regardless of whether the FAISS
        index uses IP (inner-product) or L2 distance:
          - IndexFlatIP  on unit vectors → cosine similarity directly
          - IndexFlatL2  on unit vectors → 2 - 2*cosine (same ranking order)
        """
        key = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
        with self._cache_lock:
            cached = self._embed_cache.get(key)
        if cached is not None:
            return cached

        vec = self.model.encode([text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)  # in-place L2 normalization

        with self._cache_lock:
            self._embed_cache[key] = vec
        return vec

    def clear_turn_cache(self):
        """Clear per-turn embedding cache. Should be called at the start of each new conversation turn."""
        with self._cache_lock:
            self._embed_cache.clear()

    # =========================================================================
    # CORE SEARCH (A + B + E)
    # =========================================================================

    def _search_vec(
        self,
        query_vec: np.ndarray,
        k: int,
        index: Optional[faiss.Index],
        metadata: List[Dict[str, Any]],
        index_name: str,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Inner search using a pre-computed normalized embedding.
        Handles both IP (cosine) and L2 indexes with appropriate thresholds.
        Fetches 3x candidates to allow for score filtering.
        """
        if not index:
            return []

        use_ip = _is_ip_index(index)
        if score_threshold is None:
            score_threshold = _IP_THRESHOLD if use_ip else _L2_THRESHOLD

        try:
            fetch_k = min(k * 3, max(k, index.ntotal))
            distances, ids = index.search(query_vec, fetch_k)

            results = []
            for i, doc_id in enumerate(ids[0]):
                if doc_id < 0:
                    continue
                score = float(distances[0][i])

                # (B) Filter by relevance threshold
                if use_ip and score < score_threshold:
                    continue
                if not use_ip and score > score_threshold:
                    continue

                if 0 <= doc_id < len(metadata):
                    res = metadata[doc_id].copy()
                    res["score"] = score
                    results.append(res)

                if len(results) >= k:
                    break

            return results
        except Exception as e:
            print(f"[RagPipeline] Search ERROR in {index_name}: {e}")
            return [{"source": "error", "text": f"Search error: {str(e)}"}]

    def _search(
        self,
        query_text: str,
        k: int,
        index: Optional[faiss.Index],
        metadata: List[Dict[str, Any]],
        index_name: str,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Full search pipeline: validate → embed (cached) → search_vec."""
        if not self.model:
            return [{"source": "error", "text": "RAG disabled: Embedding model failed to load."}]
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return []
        if not index:
            return []

        query_vec = self._embed(query_text)
        return self._search_vec(query_vec, k, index, metadata, index_name, score_threshold)

    # =========================================================================
    # PARALLEL MULTI-INDEX SEARCH (L + Time optimization)
    # =========================================================================

    def search_combined(
        self,
        query_text: str,
        indexes: Optional[List[str]] = None,
        k_per_index: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query ONCE then search multiple FAISS indexes in parallel.
        Returns merged, relevance-sorted results.

        Args:
            query_text:   The search query (will be embedded once and cached).
            indexes:      Which indexes to search. Options: "therapy", "therapist",
                          "personality", "legal". Default: ["therapy", "therapist"].
            k_per_index:  How many results to retrieve from each index.
        """
        if not self.model or not query_text.strip():
            return []

        all_indexes = {
            "therapy":     (self.therapy_index,      self.therapy_metadata,      "TherapyStyle"),
            "therapist":   (self.therapist_index,     self.therapist_metadata,    "Therapist"),
            "personality": (self.personality_index,   self.personality_metadata,  "Personality"),
            "legal":       (self.legal_review_index,  self.legal_review_metadata, "LegalReview"),
        }

        chosen = indexes if indexes is not None else ["therapy", "therapist"]
        tasks = [
            (name, idx, meta, iname)
            for name in chosen
            if name in all_indexes
            for idx, meta, iname in [all_indexes[name]]
            if idx is not None
        ]

        if not tasks:
            return []

        # Single encode → parallel fan-out across all indexes
        query_vec = self._embed(query_text)

        combined: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as exe:
            futures = {
                exe.submit(self._search_vec, query_vec, k_per_index, idx, meta, iname): iname
                for _, idx, meta, iname in tasks
            }
            for future in futures:
                try:
                    combined.extend(future.result())
                except Exception as e:
                    print(f"[RagPipeline] Parallel search error: {e}")

        # For IP indexes: higher score = better. For L2: lower = better.
        # Since all embeddings are normalized, use score descending (IP-style assumed after rebuild).
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)
        return combined

    # =========================================================================
    # PUBLIC SEARCH METHODS (individual index access)
    # =========================================================================

    def search_therapy_style(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        return self._search(query_text, k, self.therapy_index, self.therapy_metadata, "TherapyStyle")

    def search_personality(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._search(query_text, k, self.personality_index, self.personality_metadata, "Personality")

    def search_legal_review(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        return self._search(query_text, k, self.legal_review_index, self.legal_review_metadata, "LegalReview")

    def search_therapist(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._search(query_text, k, self.therapist_index, self.therapist_metadata, "Therapist")

    # =========================================================================
    # UTILITIES (M + J)
    # =========================================================================

    @staticmethod
    def dynamic_k(text: str, risk_score: float = 0.0) -> int:
        """
        Compute retrieval depth k based on message length and risk level.
          - High risk (>= 0.8) or long messages (> 60 words) → k = 7
          - Medium messages (30–60 words)                    → k = 5
          - Short/simple messages                            → k = 3
        """
        words = len((text or "").split())
        if risk_score >= 0.8 or words > 60:
            return 7
        if words > 30:
            return 5
        return 3

    @staticmethod
    def deduplicate_snippets(snippets: List[str], max_chars: int = 2000) -> List[str]:
        """
        Remove near-duplicate snippets and enforce a total character budget.

        Two snippets are considered duplicates if their first 60 characters
        overlap. Stops adding snippets once the total budget is exceeded.
        """
        seen_prefixes: List[str] = []
        deduped: List[str] = []
        total = 0

        for s in snippets:
            if not s.strip():
                continue
            prefix = s[:80].lower().strip()
            if any(
                prefix.startswith(p[:60]) or p.startswith(prefix[:60])
                for p in seen_prefixes
            ):
                continue
            seen_prefixes.append(prefix)
            if total + len(s) > max_chars:
                break
            deduped.append(s)
            total += len(s)

        return deduped
