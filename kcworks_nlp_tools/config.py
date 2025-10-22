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
