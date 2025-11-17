import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from traceback import format_exc

import fitz
import pytesseract
import requests
from PIL import Image
from tika import config as tika_config
from tika import initVM, language, parser

import kcworks_nlp_tools.config as config

# from kcworks_nlp_tools.dependencies import download_tika_binary
from kcworks_nlp_tools.database.db import get_db
from kcworks_nlp_tools.logging_config import set_up_logging
from kcworks_nlp_tools.models import TextExtract

# Make sure large text field in csv can be processed
csv.field_size_limit(sys.maxsize)


class DocumentExtractor:
    """
    Extract text from a file based on its extension.
    """

    def __init__(self, config=config):
        self.config = config
        self.extraction_methods = {
            ".pdf": self.extract_text_from_pdf,
            ".docx": self.extract_with_tika,
            ".pptx": self.extract_with_tika,
            ".txt": self.extract_text_from_txt,
            ".doc": self.extract_with_tika,
            ".ppt": self.extract_with_tika,
        }
        self.local_file_folder = self.config.DOWNLOADED_FILES_PATH
        self.output_file_folder = self.config.OUTPUT_FILES_PATH
        self.extracted_text_csv_path = self.config.EXTRACTED_TEXT_CSV_PATH
        self.chunk_size = int(self.config.CHUNK_SIZE)

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing newlines, carriage returns, and tabs.
        """
        return text.replace("\n", "").replace("\r", "").replace("\t", "")

    def extract_file(
        self, local_file_path: str, file_name: str
    ) -> list[tuple[str | None, int, bool, bool]]:
        """
        Extract text from a file based on its extension.

        Args:
            local_file_path (str): The path to the file.
            file_name (str): The name of the file.

        Returns:
            A list of tuples, one per chunk, containing:
            - Extracted text (or None).
            - An integer indicating the page number on which the chunk was found,
              or 0 if the file is not a PDF.
            - Flag indicating if extraction failed (True if failed, False otherwise)
            - Flag indicating if file is supported (True if supported, False otherwise)
        """
        logging.info(f"Extracting text from {file_name}")

        file_extension = os.path.splitext(file_name)[1].lower()
        language_code = language.from_file(local_file_path)
        logging.info(f"Language code detected by Tika: {language_code}")

        extraction_function = self.extraction_methods.get(file_extension)

        logging.info(f"File extension detected: {file_extension}")

        if extraction_function:
            try:
                logging.info(
                    f"Extracting {file_name} using {extraction_function.__name__}"
                )
                extracted_text = extraction_function(local_file_path)
                if any(t for t in extracted_text if t[1]):
                    return [(e[1], e[0], False, True) for e in extracted_text]
                else:
                    return [(None, 0, True, True)]
            except Exception as e:
                logging.error(f"Error extracting {file_name} at {local_file_path}: {e}")
                return [(None, 0, True, False)]
        else:
            logging.warning(f"Unsupported file type for {file_name}. Skipping.")
            return [(None, 0, True, False)]

    def extract_with_tika(self, file_path: str) -> list[tuple[int, str]] | None:
        """
        Extract text from a file using Tika and chunk it appropriately for
        transformer models.

        The text is split into chunks that:
        1. Don't cut words in the middle
        2. Don't cut sentences in the middle
        3. Are appropriate for transformer model input

        Args:
            file_path (str): The path to the file to extract text from.

        Returns:
            list[tuple[int, str]] | None: The extracted text divided into chunks as
            a list of tuples, where the first element is the page number and the
            second element is the chunk of text. If no text is found, the second
            element is None.
        """
        try:
            parsed = parser.from_file(file_path)
            content = parsed.get("content")
            if not content:
                return None

            # Split into sentences first
            sentences = [s.strip() for s in content.split(".") if s.strip()]

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())

                # If adding this sentence would exceed chunk size, start a new chunk
                if current_length + sentence_length > self.chunk_size:
                    if current_chunk:  # Only add non-empty chunks
                        # Add the page number to the chunk, even if we don't have it
                        chunks.append((0, self._clean_text(" ".join(current_chunk))))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append((0, self._clean_text(" ".join(current_chunk))))

            return chunks
        except Exception as e:
            logging.error(f"Error extracting {file_path} with Tika: {e}")
            return None

    def extract_text_from_pdf(
        self, pdf_file_path: str
    ) -> list[tuple[int, str | None]] | None:
        """
        Extract text from a PDF file with improved multi-language support
        and sentence-aware chunking.

        Args:
            pdf_file_path (str): The path to the PDF file.

        Returns:
            list[tuple[int, str | None]] | None: The extracted text divided into chunks as a list of tuples, where the first element is the page number and the second element is the chunk of text. If no text is found, the second element is None.
        """
        try:
            # First detect the language using Tika
            language_code = language.from_file(pdf_file_path)
            logging.info(f"Language code detected: {language_code}")

            with fitz.open(pdf_file_path) as doc:
                total_pages = len(doc)
                chunks = []

                for page_num in range(total_pages):
                    if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                        logging.info(f"Processing page {page_num + 1} of {total_pages}")

                    page = doc.load_page(page_num)

                    try:
                        page_text = page.get_text().strip()
                        if page_text:  # Text-based PDF
                            logging.debug(f"Page {page_num + 1}: Text-based")
                        else:  # Image-based PDF
                            logging.debug(f"Page {page_num + 1}: Image-based")
                            pix = page.get_pixmap()
                            img = Image.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples
                            )

                            # Configure PyTesseract for the detected language
                            if language_code:
                                tesseract_lang = self._convert_language_code(
                                    language_code
                                )
                                page_text = pytesseract.image_to_string(
                                    img, lang=tesseract_lang
                                ).strip()
                            else:
                                page_text = pytesseract.image_to_string(img).strip()

                            if len(page_text) < 10:
                                logging.warning(
                                    f"OCR results on page {page_num + 1} may "
                                    f"be incomplete: `{page_text}`"
                                )

                        # Split into sentences first
                        sentences = [
                            s.strip() for s in page_text.split(".") if s.strip()
                        ]

                        current_chunk = []
                        current_length = 0

                        for sentence in sentences:
                            sentence_length = len(sentence.split())

                            # If adding this sentence would exceed chunk size, start
                            # a new chunk
                            if current_length + sentence_length > self.chunk_size:
                                if current_chunk:  # Only add non-empty chunks
                                    chunks.append(
                                        self._clean_text(" ".join(current_chunk))
                                    )
                                current_chunk = [sentence]
                                current_length = sentence_length
                            else:
                                current_chunk.append(sentence)
                                current_length += sentence_length

                        # Add the last chunk if it exists
                        if current_chunk:
                            chunks.append((
                                page_num + 1,
                                self._clean_text(" ".join(current_chunk)),
                            ))

                    except Exception as page_error:
                        logging.error(
                            f"Error processing page {page_num + 1}: {page_error}"
                        )
                        chunks.append((page_num + 1, None))

                return chunks

        except Exception as e:
            logging.error(f"Error processing PDF {pdf_file_path}: {e}")
            return None

    def _convert_language_code(self, tika_lang_code: str) -> str:
        """
        Convert Tika language code to Tesseract language code format.
        """
        lang_mapping = {
            "en": "eng",
            "es": "spa",
            "fr": "fra",
            "de": "deu",
            "it": "ita",
            "pt": "por",
            "ru": "rus",
            "zh": "chi_sim",  # Simplified Chinese
            "ja": "jpn",
            "ko": "kor",
            # Add more mappings as needed
        }
        return lang_mapping.get(tika_lang_code, "eng")  # Default to English if unknown

    def extract_text_from_txt(self, txt_file_path):
        """
        Extract text from a plain txt file and chunk it appropriately for transformer models.

        Args:
            txt_file_path (str): The path to the txt file.

        Returns:
            list[str] | None: The extracted text divided into chunks as a list of strings, or None if it fails.
        """
        try:
            with open(txt_file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                logging.warning(f"Warning: The file {txt_file_path} is empty.")
                return None

            # Split into sentences first
            sentences = [s.strip() for s in text.split(".") if s.strip()]

            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())

                # If adding this sentence would exceed chunk size, start a new chunk
                if current_length + sentence_length > self.chunk_size:
                    if current_chunk:  # Only add non-empty chunks
                        chunks.append(self._clean_text(" ".join(current_chunk)))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(self._clean_text(" ".join(current_chunk)))

            return chunks

        except Exception as e:
            logging.error(f"Error extracting text from TXT file: {e}")
            return None


class KCWorksCorpusExtractor:
    """
    Access files from an API, download them, extract text, and log results.
    """

    def __init__(
        self,
        config: object = config,
        use_unique_filenames: bool = True,
        storage: str = "db",
    ):
        self.config = config
        self.use_unique_filenames = use_unique_filenames

        set_up_logging()
        # download_tika_binary()
        initVM()
        # print(tika_config.getParsers())
        # print(tika_config.getMimeTypes())
        # print(tika_config.getDetectors())
        # Make sure the working directories exist
        os.makedirs(self.config.DOWNLOADED_FILES_PATH, exist_ok=True)
        os.makedirs(self.config.OUTPUT_FILES_PATH, exist_ok=True)
        self.failed_downloads = []
        self.storage = storage

    def _get_output_path(self) -> Path:
        """
        Get the output file path, either using the base config path or generating
        a unique filename with date and counter.

        Returns:
            Path: The path to use for the output file
        """
        if not self.use_unique_filenames:
            return Path(self.config.EXTRACTED_TEXT_CSV_PATH)

        base_path = Path(self.config.EXTRACTED_TEXT_CSV_PATH)
        date_str = datetime.now().strftime("%Y%m%d")
        counter = 1
        while True:
            output_path = (
                base_path.parent
                / f"{base_path.stem}_{date_str}_{counter:03d}{base_path.suffix}"
            )
            if not output_path.exists():
                return output_path
            counter += 1

    def download_file(
        self,
        file_name: str,
        record_id: str,
        local_filename: str,
        auth_headers: dict,
        max_retries: int = 5,
        backoff_factor: int = 1,
        max_docs: int = 10,
    ) -> str | None:
        """
        Downloads a file from the specified URL and saves it locally.

        Args:
            url (str): The URL of the file to download.
            local_filename (str): The path where the file will be saved.
            auth_headers (dict): HTTP headers containing
                authentication information.
            max_retries (int, optional): Maximum number of retry attempts.
                5 is the default value.
            backoff_factor (int, optional): Delay factor for retries.
                1 is the default value.
        Returns:
            str: The path to the downloaded file, None if the download fails.
        """
        url = (
            f"{self.config.KCWORKS_API_URL}/{self.config.API_ENDPOINT}/{record_id}/"
            f"files/{file_name}/content"
        )
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to download {url} (Attempt {attempt + 1})")
                auth_headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # noqa
                )
                print(auth_headers)
                response = requests.get(url, headers=auth_headers, allow_redirects=True)
                response.raise_for_status()

                # KCWorks may provide redirect URL for file content
                if "http" in response.text:
                    file_url = response.text.strip()
                    logging.info(f"Fetching file content for title: {local_filename}")
                    file_response = requests.get(file_url, allow_redirects=True)
                    file_response.raise_for_status()

                    with open(local_filename, "wb") as file:
                        for chunk in file_response.iter_content(
                            chunk_size=8192
                        ):  # Reads the response body in chunks of 8 KB
                            file.write(chunk)
                else:
                    logging.info(
                        f"No redirect URL to fetch file content for "
                        f"title: {local_filename}"
                    )
                    with open(local_filename, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)

                logging.info(f"Successfully downloaded {local_filename}")
                return local_filename

            except requests.exceptions.RequestException as e:
                logging.error(f"Error on attempt {attempt + 1}: {e}")
                time.sleep(backoff_factor * (2**attempt))

        logging.error(f"Failed to download {url} after {max_retries} attempts")
        return None

    def _make_csv_row(
        self,
        record: dict,
        file_name: str,
        file_hash: str | None = None,
        file_type: str | None = None,
        page_num: int = 0,
        overlap: int = 0,
        extracted_text: str | None = None,
        failed: bool = False,
        supported: bool = True,
        error_msg: str = "",
    ):
        """
        Make a CSV row for the extracted text.

        Args:
            record (dict): The record from the API.
            file_name (str): The name of the file.
            file_hash (str): The file hash.
            file_type (str): The file type.
            page_num (int): The page number. This will be 0 if the text is not
                page-based. Currently used only for pdf files.
            overlap (int): The amount of overlap (in characters) between the 
                current extract and any neighbours. Defaults to 0.
            extracted_text (str): The extracted text.
            failed (bool): Whether the extraction failed. Default is False.
            supported (bool): Whether the file is supported. Default is True.
            error_msg (str): Any error mesage. Defaults to an empty string.

        Column headers:
            DOI, Record ID, File Name, File Type, File Hash, Languages, Page, 
            Overlap, Failed, Supported, Extracted Text, Error Message

        Returns:
            list: A list of the CSV row.
        """
        record_id = record["id"]
        doi = record.get("pids", {}).get("doi", {}).get("identifier", "N/A")
        languages = ",".join([
            lang["id"] for lang in record["metadata"].get("languages", [])
        ])
        if extracted_text is None:
            extracted_text = "[Error: Extraction failed]"
        failed_value = 1 if failed else 0
        supported_value = 1 if supported else 0

        return [
            doi,
            record_id,
            file_name,
            file_type,
            file_hash,
            languages,
            page,
            overlap,
            failed_value,
            supported_value,
            extracted_text,
            error_msg,
        ]

    def _write_to_csv(
        self,
        record: dict,
        file_name: str,
        extracted_text: str,
        page_num: int,
        file_type: str,
        file_hash: str | None = None,
        overlap: int | None = None,
        supported=True,
        failed=False,
        msg: str | None = None,
    ) -> None:
        """Write a csv row."""

        output_path = self._get_output_path()

        output_csv_headers = [
            "DOI",
            "Record ID",
            "File Name",
            "File Hash",
            "Languages",
            "File Type",
            "Page",
            "Overlap",
            "Supported",
            "Failed",
            "Extracted Text",
            "Error Message",
        ]

        if not Path(output_path).is_file():
            logging.info(f"Creating CSV file in {output_path}")
            try:
                with open(output_path, "w", newline="", encoding="utf-8") as file:
                    # Using tab as delimiter since extracted text may contain commas
                    # and other special characters
                    csv.writer(file, delimiter="\t").writerow(output_csv_headers)
                logging.info("Successfully created CSV file")
            except Exception as e:
                logging.error(f"Error creating CSV file: {e}")
                raise

        if failed and not supported:
            with open(output_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter="\t")
                writer.writerow(
                    self._make_csv_row(
                        record,
                        file_name,
                        file_hash,
                        file_type,
                        page_num,
                        overlap,
                        "[Download Failed]",
                        failed=True,
                        supported=False,
                        error_msg=error_msg,
                    )
                )
                self.failed_downloads.append({
                    "record": record["id"],
                    "file": file_name,
                })

    def _write_to_db(
        self,
        record: dict,
        file_name: str,
        extracted_text: str,
        page_num: int,
        file_type: str,
        file_hash: str | None = None,
        overlap: int | None = None,
        supported: bool = True,
        failed: bool = False,
        msg: str | None = None,
    ) -> None:
        """Write a db row for one extracted text section."""

        record_id = record["id"]
        doi = record.get("pids", {}).get("doi", {}).get("identifier", "N/A")
        languages = json.dumps([
            lang["id"] for lang in record["metadata"].get("languages", [])
        ])
        if extracted_text is None and msg is None:
            msg = "[Error: Extraction failed]"

        with get_db() as db:
            db.add(TextExtract(
                doi=doi,
                record_id=record_id,
                filename=file_name,
                file_hash=file_hash,
                languages=languages,
                file_type=file_type,
                page=page_num,
                overlap=overlap,
                extracted=extracted_text or msg,
                supported=supported,
                failed=failed,
                error_message=msg,
            ))

    def write_to_storage(self, *args, **kwargs) -> None:
        """Write an item to configured storage type.

        Provides a generic interface to multiple storage methods.
        """

        if self.storage == "csv":
            return self._write_to_csv(*args, **kwargs)
        elif self.storage == "db":
            return self._write_to_db(*args, **kwargs)
        else:
            raise RuntimeError("No storage type configured.")

    def extract_documents(
        self,
        max_docs: int = 0,
        verbose: bool = False,
        overwrite: bool = False,
        chunk_size: int = 500,
    ):
        """
        Access files from an API, download them, extract text, and log results.

        Args:
            max_docs (int, optional): Maximum number of documents to download.
                0 is the default value, which means the "CORPUS_SIZE" config
                value will be used.
            verbose (bool): Whether to print verbose output. Default is False.
            overwrite (bool): Whether to overwrite the already-extracted text chunks
                if they already exist.
        """
        corpus_size = max_docs if max_docs else getattr(self.config, "CORPUS_SIZE", 100)

        try:
            logging.info("Starting extraction")
            extractor = DocumentExtractor(config=self.config)
            auth_headers = {"Authorization": f"Bearer {self.config.KCWORKS_API_KEY}"}
            page = 1
            has_more_pages = True
            total_docs_processed = 0
            total_chunks_processed = 0
            total_docs_succeeded = 0
            total_chunks_succeeded = 0
            docs_with_errors = []
            failed_downloads = []

            while has_more_pages and total_docs_processed < corpus_size:
                try:
                    response = requests.get(
                        f"{self.config.KCWORKS_API_URL}/{self.config.API_ENDPOINT}?"
                        f"size={self.config.BATCH_SIZE}&page={page}",
                        headers=auth_headers,
                    )
                    response.raise_for_status()  # for non-200 responses
                    data = response.json()

                    # Process each record on the current page
                    for record in data.get("hits", {}).get("hits", []):
                        logging.info(f"Processing record {record['id']}")

                        files = record.get("files", {}).get("entries", {})
                        for file_name in files.keys():
                            file_type = files[file_name].get("ext")
                            file_hash = files[file_name].get("checksum")
                            total_docs_processed += 1
                            logging.info(
                                f"Processing file #{total_docs_processed}: {file_name}"
                            )

                            local_file_path = os.path.join(
                                self.config.DOWNLOADED_FILES_PATH, file_name
                            )

                            download_result = self.download_file(
                                file_name, record["id"], local_file_path, auth_headers
                            )
                            if not download_result:
                                logging.error(
                                    f"Failed downloading {file_name} "
                                    f"for record {record['id']}."
                                )
                                self.write_to_storage(
                                    record, 
                                    file_name, 
                                    extracted_text=None, 
                                    page_num=0,
                                    file_type=file_type,
                                    file_hash=None,
                                    overlap=0,
                                    supported=False, 
                                    failed=True, 
                                    msg="[Error: Download failed]"
                                )
                                continue

                            try:
                                extracted_text = extractor.extract_file(
                                    local_file_path, file_name
                                )
                                chunks_succeeded = 0
                                chunks_failed = 0
                                chunks_unsupported = 0
                                for txt, pg, failed, supported in extracted_text:
                                    total_chunks_processed += 1
                                    if supported and not failed:
                                        chunks_succeeded += 1
                                    elif not supported:
                                        chunks_failed += 1
                                        chunks_unsupported += 1
                                    elif failed:
                                        chunks_failed += 1
                                if chunks_failed == chunks_unsupported == 0:
                                    total_chunks_succeeded += chunks_succeeded
                                    total_docs_succeeded += 1
                                else:
                                    docs_with_errors.append({
                                        "record": record["id"],
                                        "file": file_name,
                                        "chunks_succeeded": chunks_succeeded,
                                        "chunks_failed": chunks_failed,
                                        "chunks_unsupported": chunks_unsupported,
                                    })

                            except Exception as e:
                                logging.error(
                                    f"Error extracting file {file_name} "
                                    f"for record {record['id']}: {format_exc(e)}."
                                )
                                self.write_to_storage(
                                    record, 
                                    file_name, 
                                    extracted_text=None, 
                                    page_num=0,
                                    file_type=file_type,
                                    file_hash=file_hash,
                                    overlap=0,
                                    supported=False, 
                                    failed=True, 
                                    msg="[Processing error during extraction]"
                                )
                                docs_with_errors.append({
                                    "record": record["id"],
                                    "file": file_name,
                                })
                                continue

                            # Append results to the output storage
                            for chunk in extracted_text if chunk and not chunk[2]:
                                self.write_to_storage(
                                    record,
                                    file_name,
                                    extracted_text=chunk[1],
                                    page_num=chunk[0],
                                    file_type=file_type,
                                    file_hash=file_hash,
                                    overlap=0,  # FIXME: Get this value
                                    supported=chunk[3],
                                    failed=chunk[2],
                                    msg="",
                                )

                            # If successfully processed, delete the file
                            if extracted_text and not any(
                                chunk for chunk in extracted_text if chunk and chunk[2]
                            ):
                                os.remove(local_file_path)
                                logging.info(f"Deleted local file: {file_name}")
                            else:
                                logging.warning(
                                    f"Failed to process {file_name} for {record['id']}."
                                )

                    # Check if there are more pages to fetch
                    logging.info(f"Completed page {page}.")
                    has_more_pages = "next" in data.get("links", {})
                    page += 1

                except requests.exceptions.HTTPError as e:
                    if (
                        e.response
                        and e.response.status_code == 400
                        and f"page={page}" in str(e)
                    ):
                        logging.info(
                            f"Reached the Invenio page limit at page {page}. "
                            "Accessible files processed and results written to CSV."
                        )
                        break
                    else:
                        logging.error(f"HTTP error: {e}")
                except Exception as e:
                    logging.error(f"An error occurred: {e}")

            logging.info("===================================================")
            logging.info("Finished processing")
            logging.info("===================================================")
            logging.info(f"attempted corpus size: {corpus_size}")
            logging.info(f"documents processed: {total_docs_processed}")
            logging.info(f"    chunks processed: {total_chunks_processed}")
            logging.info(f"documents succeeded: {total_docs_succeeded}")
            logging.info(f"    chunks succeeded: {total_chunks_succeeded}")
            logging.info(f"documents with errors: {len(docs_with_errors)}")
            logging.info(
                f"    chunks succeeded: {total_chunks_processed - total_chunks_succeeded}"
            )
            logging.info(f"downloads failed: {len(failed_downloads)}")
            if verbose:
                logging.info("===================================================")
                logging.info(f"documents with errors:\n{pformat(docs_with_errors)}")
                logging.info("===================================================")
                logging.info(f"failed downloads:\n{pformat(failed_downloads)}")

        finally:
            logging.info("===================================================")
            logging.info("Processing complete.")
            logging.info("===================================================")


def main(
    verbose: bool = False,
    corpus_size: int = 10,
    overwrite: bool = False,
    chunk_size: int = 500,
) -> None:
    """Download KCWorks record files and extract text."""
    KCWorksCorpusExtractor().extract_documents(
        max_docs=corpus_size,
        verbose=verbose,
        overwrite=overwrite,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract KCWorks documents into a csv file "
            "of chunked text sections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" """,
    )

    parser.add_argument(
        "--corpus-size",
        "-c",
        type=int,
        help="Number of documents to download and extract",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Re-process and replace documents that already exist in the file.",
    )
    parser.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        help="Number of characters to include in each chunk.",
    )

    args = parser.parse_args()
    main(
        verbose=args.verbose,
        corpus_size=args.corpus_size,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
    )
