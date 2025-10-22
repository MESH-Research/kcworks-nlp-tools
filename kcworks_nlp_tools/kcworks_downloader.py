import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from traceback import format_exc

import fitz
import pytesseract
import requests
from PIL import Image
from tika import config as tika_config
from tika import initVM, language, parser

import kcworks_nlp_tools.config as config
from kcworks_nlp_tools.dependencies import download_tika_binary
from kcworks_nlp_tools.logging_config import set_up_logging

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
        logging.info(f"Local file path: {local_file_path}")
        logging.info(os.getenv("TIKA_SERVER_URL"))

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

    def __init__(self, config=config, use_unique_filenames=True):
        self.config = config
        self.use_unique_filenames = use_unique_filenames

        set_up_logging()
        # download_tika_binary()
        initVM()
        print(tika_config.getParsers())
        print(tika_config.getMimeTypes())
        print(tika_config.getDetectors())
        # Make sure the working directories exist
        os.makedirs(self.config.DOWNLOADED_FILES_PATH, exist_ok=True)
        os.makedirs(self.config.OUTPUT_FILES_PATH, exist_ok=True)

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
        page_num: int,
        extracted_text: str | None,
        failed: bool = False,
        supported: bool = True,
    ):
        """
        Make a CSV row for the extracted text.

        Args:
            record (dict): The record from the API.
            file_name (str): The name of the file.
            page_num (int): The page number. This will be 0 if the text is not
                page-based. Currently used only for pdf files.
            extracted_text (str): The extracted text.
            failed (bool): Whether the extraction failed. Default is False.
            supported (bool): Whether the file is supported. Default is True.

        Column headers:
            Record ID, DOI, Languages, File Name, Extracted Text, Failed, Supported

        Returns:
            list: A list of the CSV row.
        """
        record_id = record["id"]
        doi = record.get("pids", {}).get("doi", {}).get("identifier", "N/A")
        languages = ",".join([
            lang["id"] for lang in record["metadata"].get("languages", [])
        ])
        if extracted_text is None:
            extracted_text = "[Error: Extraction Failed]"
        failed_value = 1 if failed else 0
        supported_value = 1 if supported else 0

        return [
            record_id,
            doi,
            languages,
            file_name,
            extracted_text,
            failed_value,
            supported_value,
        ]

    def extract_documents(self, max_docs: int = 0):
        """
        Access files from an API, download them, extract text, and log results.

        Args:
            max_docs (int, optional): Maximum number of documents to download.
                0 is the default value, which means the "CORPUS_SIZE" config
                value will be used.
        """
        corpus_size = max_docs if max_docs else getattr(self.config, "CORPUS_SIZE", 100)
        try:
            output_path = self._get_output_path()
            logging.info(f"Starting extraction. Output will be saved to: {output_path}")
            extractor = DocumentExtractor(config=self.config)
            auth_headers = {"Authorization": f"Bearer {self.config.KCWORKS_API_KEY}"}
            page = 1
            has_more_pages = True
            total_docs_extracted = 0

            output_csv_headers = [
                "Record ID",
                "DOI",
                "Languages",
                "File Name",
                "Extracted Text",
                "Failed",
                "Supported",
            ]

            # Create the CSV file and write headers if it's the first page
            if page == 1:
                logging.info("Creating CSV file with headers")
                try:
                    with open(output_path, "w", newline="", encoding="utf-8") as file:
                        # Using tab as delimiter since extracted text may contain commas
                        # and other special characters
                        csv.writer(file, delimiter="\t").writerow(output_csv_headers)
                    logging.info("Successfully created CSV file")
                except Exception as e:
                    logging.error(f"Error creating CSV file: {e}")
                    raise

            logging.info(
                f"About to enter while loop with page={page}, has_more_pages={has_more_pages}, corpus_size={corpus_size}"
            )
            while has_more_pages and total_docs_extracted < corpus_size:
                logging.info(f"Inside while loop, page={page}")
                try:
                    logging.info(f"Making API request for page {page}")
                    response = requests.get(
                        f"{self.config.KCWORKS_API_URL}/{self.config.API_ENDPOINT}?"
                        f"size={self.config.BATCH_SIZE}&page={page}",
                        headers=auth_headers,
                    )
                    response.raise_for_status()  # for non-200 responses
                    data = response.json()
                    logging.info(f"Got API response: {data}")

                    # Process each record on the current page
                    for record in data.get("hits", {}).get("hits", []):
                        logging.info(f"Processing record {record['id']}")

                        files = record.get("files", {}).get("entries", {})
                        for file_name in files.keys():
                            total_docs_extracted += 1
                            logging.info(
                                f"Processing file #{total_docs_extracted}: {file_name}"
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
                                with open(
                                    output_path, "a", newline="", encoding="utf-8"
                                ) as file:
                                    # Using tab as delimiter since extracted text
                                    # may contain commas
                                    # and other special characters
                                    writer = csv.writer(file, delimiter="\t")
                                    writer.writerow(
                                        self._make_csv_row(
                                            record,
                                            file_name,
                                            0,
                                            "[Download Failed]",
                                            failed=True,
                                        )
                                    )
                                continue

                            try:
                                extracted_text = extractor.extract_file(
                                    local_file_path, file_name
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error extracting file {file_name} "
                                    f"for record {record['id']}: {format_exc(e)}."
                                )
                                with open(
                                    output_path, "a", newline="", encoding="utf-8"
                                ) as file:
                                    # Using tab as delimiter since extracted text
                                    # may contain commas
                                    # and other special characters
                                    writer = csv.writer(file, delimiter="\t")
                                    writer.writerow(
                                        self._make_csv_row(
                                            record,
                                            file_name,
                                            0,
                                            "[Processing Error]",
                                            failed=True,
                                        )
                                    )
                                continue

                            # Append results to the output CSV
                            with open(
                                output_path, "a", newline="", encoding="utf-8"
                            ) as file:
                                # Using tab as delimiter since extracted text
                                # may contain commas
                                # and other special characters
                                writer = csv.writer(file, delimiter="\t")
                                if isinstance(extracted_text, list):
                                    for chunk in extracted_text:
                                        writer.writerow(
                                            self._make_csv_row(
                                                record,
                                                file_name,
                                                chunk[1],
                                                chunk[0],
                                                failed=chunk[2],
                                                supported=chunk[3],
                                            )
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

        finally:
            logging.info("Processing complete.")


def main() -> None:
    """Download KCWorks record files and extract text."""
    KCWorksCorpusExtractor().extract_documents()


if __name__ == "__main__":
    main()
