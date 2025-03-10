import os
from pprint import pformat
import time
import logging
from traceback import format_exc
import requests
import fitz
from pathlib import Path
import pytesseract
from PIL import Image

# from docx import Document
# import pptx
import sys
import csv
from tika import language, parser, initVM
from tika import config as tika_config

# from .dependencies import download_tika_binary
from logging_config import set_up_logging
import config


# Make sure large text field in csv can be processed
csv.field_size_limit(sys.maxsize)


class DocumentExtractor:
    """
    Extract text from a file based on its extension.
    """

    def __init__(self):
        self.extraction_methods = {
            ".pdf": self.extract_text_from_pdf,
            ".docx": self.extract_with_tika,
            ".pptx": self.extract_with_tika,
            ".txt": self.extract_text_from_txt,
            ".doc": self.extract_with_tika,
            ".ppt": self.extract_with_tika,
        }
        self.local_file_folder = config.DOWNLOADED_FILES_PATH
        self.output_file_folder = config.OUTPUT_FILES_PATH
        self.extracted_text_csv_path = config.EXTRACTED_TEXT_CSV_PATH
        self.chunk_size = config.CHUNK_SIZE

    def extract_file(
        self, local_file_path: str, file_name: str
    ) -> tuple[str | None, bool, bool]:
        """
        Extract text from a file based on its extension.

        Args:
            local_file_path (str): The path to the file.
            file_name (str): The name of the file.

        Returns:
            A tuple containing:
            - Extracted text (or None). If the file is a PDF, the text is a dictionary
              whose keys are the page numbers and values are lists of strings, each
              representing a chunk of text from that page.
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
                if extracted_text:
                    return (extracted_text, False, True)
                else:
                    return (None, True, True)
            except Exception as e:
                logging.error(f"Error extracting {file_name} at {local_file_path}: {e}")
                return (None, True, True)
        else:
            logging.warning(f"Unsupported file type for {file_name}. Skipping.")
            return (None, True, False)

    def extract_with_tika(self, file_path: str) -> str | None:
        """
        Extract text from a file using Tika.
        """
        try:
            parsed = parser.from_file(file_path)
            words = parsed.get("content").split()
            text_chunks = [
                " ".join(words[i : i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size)
            ]
            return text_chunks
        except Exception as e:
            logging.error(f"Error extracting {file_path} with Tika: {e}")
            return None

    def extract_text_from_pdf(self, pdf_file_path: str) -> dict[str, str | None] | None:
        """
        Extract text from a PDF file.
        Uses PyMuPDF/fitz for text-based PDFs and PyTesseract for image-based PDFs.
        Divides the pages into chunks of a specified size when extracting text.

        # TODO: think about how to avoid losing semantic information when
        # dividing sentences at page and chunk boundaries

        Args:
            pdf_file_path (str): The path to the PDF file.
            chunk_size (int, optional): The number of words to include in each chunk
            when dividing the pages into chunks.

        Returns:
            A dictionary whose keys are the page numbers and values are the
            extracted text.
        """
        # Read PDF file
        # FIXME: Alternative approach using Tika
        # data = parser.from_file(filename, xmlContent=True)
        # xhtml_data = BeautifulSoup(data['content'])
        # for i, content in enumerate(xhtml_data.find_all('div', attrs={'class': 'page'})):
        #     # Parse PDF data using TIKA (xml/html)
        #     # It's faster and safer to create a new buffer than truncating it
        #     # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
        #     _buffer = StringIO()
        #     _buffer.write(str(content))
        #     parsed_content = parser.from_buffer(_buffer.getvalue())

        #     # Add pages
        #     text = parsed_content['content'].strip()
        #     pages_txt.append(text)

        try:
            with fitz.open(pdf_file_path) as doc:
                total_pages = len(doc)
                page_texts = {}
                for page_num in range(total_pages):
                    # Log every 10 pages or the last page
                    if (page_num + 1) % 10 == 0 or page_num == total_pages - 1:
                        logging.info(f"Processing page {page_num + 1} of {total_pages}")

                    page = doc.load_page(page_num)

                    try:
                        page_text = page.get_text().strip()
                        if page_text:  # If there is text, pdf likely text-based
                            logging.debug(f"Page {page_num + 1}: Text-based")
                        else:  # This is probably image-based
                            logging.debug(f"Page {page_num + 1}: Image-based")
                            pix = page.get_pixmap()
                            img = Image.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples
                            )
                            page_text = pytesseract.image_to_string(img).strip()

                            if len(page_text) < 10:
                                logging.warning(
                                    f"OCR results on page {page_num + 1} may "
                                    f"be incomplete: `{page_text}`"
                                )

                        chunks = []
                        chunk_count = len(page_text) // self.chunk_size
                        for i in range(chunk_count):
                            chunk = page_text[
                                i * self.chunk_size : (i + 1) * self.chunk_size
                            ]
                            chunks.append(chunk)
                        page_texts[page_num + 1] = chunks

                    except Exception as page_error:
                        logging.error(
                            f"Error processing page {page_num + 1}: {page_error}"
                        )
                        page_texts[page_num + 1] = [None]

                return page_texts

        except Exception as e:
            logging.error(f"Error processing PDF {pdf_file_path}: {e}")
            return None

    def extract_text_from_docx(self, docx_file_path: str) -> list[str] | None:
        """
        Extract text from a .docx file.

        Args:
            docx_file_path (str): The path to the .docx file.

        Returns:
            list[str]: The extracted text divided into a list of strings, or None
            if it fails.
        """
        try:
            logging.info(f"Opening DOCX file: {docx_file_path}")
            doc = Document(docx_file_path)
            paragraphs = []

            # Loop through each paragraph
            for paragraph_num, paragraph in enumerate(doc.paragraphs, start=1):
                # Check if the paragraph text contains unexpected
                # formatting or artifacts
                if paragraph.text.strip():
                    clean_text = paragraph.text.strip()
                    # TODO: python has library to take out markup instruction
                    if (
                        "<" in clean_text or ">" in clean_text
                    ):  # Check for HTML or XML or markup
                        logging.warning(
                            f"Potential formatting instructions detected in "
                            f"paragraph {paragraph_num}: {clean_text}"
                        )
                    paragraphs.append(clean_text)
                else:
                    logging.debug(
                        f"Paragraph {paragraph_num} is empty and was skipped."
                    )

            extracted_text = "\n".join(paragraphs)
            logging.info(
                f"Successfully extracted text from {len(paragraphs)} paragraphs."
            )
            return extracted_text

        except PermissionError as e:
            logging.warning(f"Permission denied for {docx_file_path}: {e}")
            return None

        except Exception as e:
            logging.error(f"Error extracting text from DOCX file {docx_file_path}: {e}")
            return None

    def extract_text_from_doc_with_tika(self, doc_file_path):
        """
        Extracts text from a .doc file using Apache Tika server.

        Args:
            doc_file_path (str): Path to the .doc file.

        Returns:
            str: Extracted text, or None if extraction fails.
        """
        try:
            parsed = parser.from_file(doc_file_path)  # Send the file to Tika server
            return parsed.get("content")  # Extract the content field (text)
        except Exception as e:
            print(f"Error extracting text from {doc_file_path}: {e}")
            return None

    def extract_text_from_ppt_with_tika(self, ppt_file_path):
        """
        Extract text from a .ppt file using Apache Tika.

        Args:
            ppt_file_path (str): The path to the .ppt file.

        Returns:
            str: Extracted text, or None if extraction fails.
        """
        try:
            parsed = parser.from_file(ppt_file_path)
            return parsed.get("content")
        except Exception as e:
            logging.error(f"Error extracting text from {ppt_file_path}: {e}")
            return None

    def extract_text_from_txt(self, txt_file_path):
        """
        Extract text from a plain txt file using Python's built-in open() function.

        Args:
            txt_file_path (str): The path to the txt file.

        Returns:
            str: The extracted text, or None if it fails.
        """
        try:
            with open(txt_file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                logging.warning(f"Warning: The file {txt_file_path} is empty.")

            return text
        except Exception as e:
            logging.error(f"Error extracting text from TXT file: {e}")
            return None

    def extract_text_from_pptx(
        self,
        pptx_file_path,
    ):  # TODO: .ppt not supported, check tika and how to integrate
        """
        Extract text from a PowerPoint file, including slide text, speaker notes,
        and tables.

        Args:
            pptx_file_path (str): The path to the PowerPoint file.

        Returns:
            str: The extracted text, or None if it fails.
        """
        try:
            logging.info(f"Opening PowerPoint file: {pptx_file_path}")
            presentation = pptx.Presentation(pptx_file_path)
            all_slides_text = []

            for slide_num, slide in enumerate(presentation.slides, start=1):
                slide_text = []

                for shape in slide.shapes:
                    # Extract text from text-containing shapes
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text.strip())

                    # Extract text from table
                    if shape.shape_type == 19:  # Type 19 matches to a table
                        for row in shape.table.rows:
                            for cell in row.cells:
                                cell_text = cell.text.strip()
                                if cell_text:
                                    slide_text.append(cell_text)

                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_text.append(f"Speaker Notes: {notes}")

                logging.info(f"Processed slide {slide_num}/{len(presentation.slides)}")

                if slide_text:
                    # Combine slide content with headers
                    all_slides_text.append(
                        f"--- Slide {slide_num} ---\n" + "\n".join(slide_text)
                    )
                else:
                    logging.warning(f"Slide {slide_num} contains no extractable text.")

            return "\n\n".join(all_slides_text)

        except Exception as e:
            logging.error(
                f"Error extracting text from PowerPoint file {pptx_file_path}: {e}"
            )
            return None


class KCWorksCorpusExtractor:
    """
    Access files from an API, download them, extract text, and log results.
    """

    def __init__(self, config=config):
        self.config = config

    def download_file(
        self,
        file_name: str,
        record_id: str,
        local_filename: str,
        auth_headers: dict,
        max_retries: int = 5,
        backoff_factor: int = 1,
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
            f"{config.KCWORKS_API_URL}/{config.API_ENDPOINT}/{record_id}/"
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
        extracted_text: str,
        failed: bool = False,
        supported: bool = True,
    ):
        """
        Make a CSV row for the extracted text.

        Args:
            record (dict): The record from the API.
            file_name (str): The name of the file.
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
        languages = ",".join(
            [lang["id"] for lang in record["metadata"].get("languages", [])]
        )
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

    def extract_documents(self):
        """
        Access files from an API, download them, extract text, and log results.
        """
        try:
            extractor = DocumentExtractor()
            auth_headers = {"Authorization": f"Bearer {config.KCWORKS_API_KEY}"}
            page = 1
            has_more_pages = True
            counter = 0

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
                with open(
                    config.EXTRACTED_TEXT_CSV_PATH, "w", newline="", encoding="utf-8"
                ) as file:
                    csv.writer(file).writerow(output_csv_headers)

            while has_more_pages:
                try:
                    response = requests.get(
                        f"{config.KCWORKS_API_URL}/{config.API_ENDPOINT}?"
                        f"size={config.BATCH_SIZE}&page={page}",
                        headers=auth_headers,
                    )
                    response.raise_for_status()  # for non-200 responses
                    data = response.json()

                    # Process each record on the current page
                    for record in data.get("hits", {}).get("hits", []):
                        logging.info(f"Processing record {record['id']}")

                        files = record.get("files", {}).get("entries", {})
                        for file_name in files.keys():
                            counter += 1
                            logging.info(f"Processing file #{counter}: {file_name}")

                            local_file_path = os.path.join(
                                config.DOWNLOADED_FILES_PATH, file_name
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
                                    config.EXTRACTED_TEXT_CSV_PATH,
                                    "a",
                                    newline="",
                                    encoding="utf-8",
                                ) as file:
                                    writer = csv.writer(file)
                                    writer.writerow(
                                        self._make_csv_row(
                                            record,
                                            file_name,
                                            "[Download Failed]",
                                            failed=True,
                                        )
                                    )
                                continue

                            try:
                                extracted_text, failed, supported = (
                                    extractor.extract_file(local_file_path, file_name)
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error extracting file {file_name} "
                                    f"for record {record['id']}: {format_exc(e)}."
                                )
                                with open(
                                    config.EXTRACTED_TEXT_CSV_PATH,
                                    "a",
                                    newline="",
                                    encoding="utf-8",
                                ) as file:
                                    writer = csv.writer(file)
                                    writer.writerow(
                                        self._make_csv_row(
                                            record,
                                            file_name,
                                            "[Processing Error]",
                                            failed=True,
                                        )
                                    )
                                continue

                            # Append results to the output CSV
                            with open(
                                config.EXTRACTED_TEXT_CSV_PATH,
                                "a",
                                newline="",
                                encoding="utf-8",
                            ) as file:
                                writer = csv.writer(file)
                                if isinstance(extracted_text, list):
                                    for chunk in extracted_text:
                                        writer.writerow(
                                            self._make_csv_row(
                                                record,
                                                file_name,
                                                chunk,
                                                failed=failed,
                                                supported=supported,
                                            )
                                        )
                                elif isinstance(extracted_text, dict):
                                    for page_num, chunks in extracted_text.items():
                                        for chunk in chunks:
                                            writer.writerow(
                                                self._make_csv_row(
                                                    record,
                                                    file_name,
                                                    chunk,
                                                    failed=failed,
                                                    supported=supported,
                                                )
                                            )
                                else:
                                    writer.writerow(
                                        self._make_csv_row(
                                            record,
                                            file_name,
                                            extracted_text,
                                            failed=failed,
                                            supported=supported,
                                        )
                                    )

                            # If successfully processed, delete the file
                            if not failed:
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
    set_up_logging()
    # download_tika_binary()
    initVM()
    print(tika_config.getParsers())
    print(tika_config.getMimeTypes())
    print(tika_config.getDetectors())
    # Make sure the working directories exist
    os.makedirs(config.DOWNLOADED_FILES_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_FILES_PATH, exist_ok=True)

    KCWorksCorpusExtractor().extract_documents()


if __name__ == "__main__":
    main()
