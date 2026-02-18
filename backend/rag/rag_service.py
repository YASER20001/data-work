import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path

# --- Constants ---
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

THERAPY_INDEX_FILE = SCRIPT_DIR / "therapy_rag_index.faiss"
RAG_METADATA_FILE = SCRIPT_DIR / "rag_metadata.json"

PERSONALITY_INDEX_FILE = SCRIPT_DIR / "personality_rag_index.faiss"
PERSONALITY_METADATA_FILE = SCRIPT_DIR / "personality_criteria.json"

LEGAL_RAG_INDEX_FILE = SCRIPT_DIR / "legal_review_index.faiss"
LEGAL_RAG_METADATA_FILE = SCRIPT_DIR / "legal_review_metadata.json"

THERAPIST_INDEX_FILE = SCRIPT_DIR / "therapist_rag_index.faiss"
THERAPIST_METADATA_FILE = SCRIPT_DIR / "therapist_rag_metadata.json"



class RagPipeline:
    """
    Centralized RAG Service.
    Initializes the embedding model and loads FAISS indexes/metadata.
    """

    def __init__(self, model_name: str = None):
        print(f"[RagPipeline] Initializing...")

        # Force default model if None is passed
        if not model_name:
            print(f"[RagPipeline] No model name provided. Defaulting to: {DEFAULT_MODEL_NAME}")
            model_name = DEFAULT_MODEL_NAME

        # 1. Load Model
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()

            # Ensure the model actually works immediately
            _ = self.model.encode(["test integrity check"])

            print(f"[RagPipeline] SUCCESS: Loaded embedding model: {model_name} (Dim: {self.dimension})")
        except Exception as e:
            print(f"\n[RagPipeline] CRITICAL ERROR: Could not load SentenceTransformer model!")
            print(f"Reason: {str(e)}")
            self.model = None
            self.dimension = 0

        # 2. Load Indexes
        self.therapy_index = self._load_index(THERAPY_INDEX_FILE, "TherapyStyle")
        self.therapy_metadata = self._load_metadata(RAG_METADATA_FILE, "TherapyStyle")

        self.personality_index = self._load_index(PERSONALITY_INDEX_FILE, "Personality")
        self.personality_metadata = self._load_metadata(PERSONALITY_METADATA_FILE, "Personality")

        self.legal_review_index = self._load_index(LEGAL_RAG_INDEX_FILE, "LegalReview")
        self.legal_review_metadata = self._load_metadata(LEGAL_RAG_METADATA_FILE, "LegalReview")

        self.therapist_index = self._load_index(THERAPIST_INDEX_FILE, "Therapist")
        self.therapist_metadata = self._load_metadata(THERAPIST_METADATA_FILE, "Therapist")

        print("[RagPipeline] Initialization complete.")

    def _load_index(self, file_path: Path, name: str) -> Optional[faiss.Index]:
        if not file_path.exists():
            print(f"[RagPipeline] WARNING: {name} index file not found at {file_path}")
            return None
        try:
            index = faiss.read_index(str(file_path))
            if self.model and index.d != self.dimension:
                print(f"[RagPipeline] WARNING: {name} index dim ({index.d}) != model dim ({self.dimension})")
            print(f"[RagPipeline] Loaded {name} index ({index.ntotal} vectors).")
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

            # Check if data is in the new "profiles" format
            if isinstance(data, dict) and "profiles" in data:
                print(f"[RagPipeline] Loaded {name} from 'profiles' key.")
                return data["profiles"]
            # Fallback for old list format
            elif isinstance(data, list):
                return data

            return []
        except Exception as e:
            print(f"[RagPipeline] ERROR: Could not load {name} metadata: {e}")
            return []

    def _search(self, query_text: str, k: int, index: Optional[faiss.Index], metadata: List[Dict[str, Any]], index_name: str) -> List[Dict[str, Any]]:
        # 1. Check Model Integrity
        if not self.model:
            return [{"source": "error", "text": "RAG disabled: Embedding model failed to load."}]

        # 2. Check Input Validity
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            return []

        # 3. Check Index Availability
        if not index:
            return []

        try:
            query_vector = self.model.encode([query_text])
            distances, ids = index.search(query_vector, k)

            results = []
            for i, doc_id in enumerate(ids[0]):
                if 0 <= doc_id < len(metadata):
                    res = metadata[doc_id].copy()
                    res["score"] = float(distances[0][i])
                    results.append(res)
            return results
        except Exception as e:
            print(f"[RagPipeline] Search ERROR in {index_name}: {e}")
            return [{"source": "error", "text": f"Search error: {str(e)}"}]

    def search_therapy_style(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        return self._search(
            query_text,
            k,
            self.therapy_index,
            self.therapy_metadata,
            "TherapyStyle",
        )

    def search_personality(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Run a RAG search over the Personality index.
        Returns a list of result dicts with 'text' and 'metadata'.
        """
        return self._search(
            query_text,
            k,
            self.personality_index,
            self.personality_metadata,
            "Personality",
        )

    def search_legal_review(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        return self._search(
            query_text,
            k,
            self.legal_review_index,
            self.legal_review_metadata,
            "LegalReview",
        )

    def search_therapist(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._search(
            query_text,
            k,
            self.therapist_index,
            self.therapist_metadata,
            "Therapist",
        )
