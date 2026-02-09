import os
from pathlib import Path
from dotenv import load_dotenv

if os.path.exists(Path(__file__).parent.parent / ".env"):
    load_dotenv(Path(__file__).parent.parent / ".env")
current_dir = Path(__file__).parent

DOWNLOADED_FILES_PATH = os.getenv(
    "DOWNLOADED_FILES_PATH", current_dir / "downloaded_files"
)
OUTPUT_FILES_PATH = os.getenv("OUTPUT_FILES_PATH", current_dir / "output_files")

API_ENDPOINT = "records"
BATCH_SIZE = os.getenv("BATCH_SIZE", 10)
CORPUS_SIZE = os.getenv("CORPUS_SIZE", 10)
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 384)
EXTRACTED_TEXT_CSV_PATH = Path(OUTPUT_FILES_PATH) / "extracted_text.csv"
KCWORKS_API_KEY = os.getenv("KCWORKS_API_KEY")
KCWORKS_API_URL = os.getenv("KCWORKS_API_URL", "https://works.hcommons.org/api")
LOGS_PATH = Path(current_dir) / "logs"
PREPROCESSED_PATH = Path(OUTPUT_FILES_PATH) / "preprocessed.csv"
TIKA_LOG_PATH = LOGS_PATH
TIKA_PATH = Path(current_dir) / "lib"
TIKA_LOG_FILE = "tika.log"

# Langchain semantic search (langchain_semantic_search_test.py)
DEFAULT_RESULT_SIZE = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Short slugs for embedding model names (DB dir names). Add an entry for every model you use.
MODEL_SLUGS: dict[str, str] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm-l6",
    "sentence-transformers/all-mpnet-base-v2": "mpnet-base",
}
CHROMA_DISTANCE_ALGORITHMS = ("l2", "cosine", "ip")
DEFAULT_DISTANCE_ALGORITHM = "l2"
CHROMA_DB_DIR_NAME = "chroma_langchain_db"
# Facets (schemes) for loading and search. One dict per facet: scheme, file, template.
# scheme = metadata "scheme" in the vector store and --facet choices.
FACETS = (
    {"scheme": "topical", "file": "data/fast_subjects/subjects_fast_topical.jsonl", "template": "the topic"},
    {"scheme": "chronological", "file": "data/fast_subjects/subjects_fast_chronological.jsonl", "template": "the chronological date or period"},
    {"scheme": "corporate", "file": "data/fast_subjects/subjects_fast_corporate.jsonl", "template": "the organization, group, or movement"},
    {"scheme": "event", "file": "data/fast_subjects/subjects_fast_event.jsonl", "template": "the historical or current event"},
    {"scheme": "formgenre", "file": "data/fast_subjects/subjects_fast_formgenre.jsonl", "template": "the media form or genre"},
    {"scheme": "geographic", "file": "data/fast_subjects/subjects_fast_geographic.jsonl", "template": "the geographic location or region"},
    {"scheme": "meeting", "file": "data/fast_subjects/subjects_fast_meeting.jsonl", "template": "the meeting, conference, or event for dialogue"},
    {"scheme": "personal", "file": "data/fast_subjects/subjects_fast_personal.jsonl", "template": "the person"},
    {"scheme": "title", "file": "data/fast_subjects/subjects_fast_title.jsonl", "template": "the title of a work"},
)
FACET_SCHEME_NAMES = tuple(f["scheme"] for f in FACETS)

# Vector store backend: "chroma" | "opensearch". Env VECTOR_STORE_BACKEND; CLI --backend overrides.
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
if VECTOR_STORE_BACKEND not in ("chroma", "opensearch"):
    VECTOR_STORE_BACKEND = "chroma"

