import pytest
import csv
import os
from unittest.mock import Mock, patch
from pathlib import Path
from kcworks_nlp_tools.kcworks_downloader import (
    KCWorksCorpusExtractor,
    DocumentExtractor,
)
import kcworks_nlp_tools.config as config


@pytest.fixture
def mock_response():
    mock = Mock()
    mock.json.return_value = {
        "hits": {
            "hits": [
                {
                    "id": "test123",
                    "pids": {"doi": {"identifier": "10.1234/test"}},
                    "metadata": {"languages": [{"id": "eng"}]},
                    "files": {"entries": {"test.pdf": {}}},
                }
            ]
        },
        "links": {},
    }
    return mock


@pytest.fixture
def mock_extractor():
    mock = Mock(spec=DocumentExtractor)
    mock.extract_file.return_value = [(1, "Extracted text", False, True)]
    return mock


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration with temporary directories."""
    downloaded_path = tmp_path / "downloaded"
    output_path = tmp_path / "output"
    os.makedirs(downloaded_path)
    os.makedirs(output_path)

    # Create a test config object
    test_config = config
    test_config.DOWNLOADED_FILES_PATH = downloaded_path
    test_config.OUTPUT_FILES_PATH = output_path
    test_config.EXTRACTED_TEXT_CSV_PATH = output_path / "extracted_text.csv"
    test_config.BATCH_SIZE = 1
    test_config.CORPUS_SIZE = 1
    test_config.CHUNK_SIZE = 400

    return test_config


def test_extract_documents(mock_response, mock_extractor, test_config):
    # Set up temporary paths
    downloaded_path = test_config.DOWNLOADED_FILES_PATH
    csv_path = test_config.EXTRACTED_TEXT_CSV_PATH

    with (
        patch(
            "kcworks_nlp_tools.kcworks_downloader.DocumentExtractor",
            return_value=mock_extractor,
        ),
        patch("requests.get", return_value=mock_response),
        patch("kcworks_nlp_tools.kcworks_downloader.config") as mock_config,
        patch("kcworks_nlp_tools.kcworks_downloader.os.remove"),
        patch("kcworks_nlp_tools.kcworks_downloader.initVM"),
        patch(
            "kcworks_nlp_tools.kcworks_downloader.tika_config.getParsers",
            return_value=[],
        ),
        patch(
            "kcworks_nlp_tools.kcworks_downloader.tika_config.getMimeTypes",
            return_value=[],
        ),
        patch(
            "kcworks_nlp_tools.kcworks_downloader.tika_config.getDetectors",
            return_value=[],
        ),
    ):
        # Configure mock config
        mock_config = test_config
        mock_config.KCWORKS_API_KEY = "test_key"
        mock_config.KCWORKS_API_URL = "http://test.com"
        mock_config.BATCH_SIZE = 10
        mock_config.CORPUS_SIZE = 1

        # Create downloader instance and patch its download_file method
        downloader = KCWorksCorpusExtractor(config=mock_config)
        downloader.download_file = Mock(return_value=str(downloaded_path / "test.pdf"))

        # Run extraction
        downloader.extract_documents(max_docs=1)

        # Verify CSV was created with correct content
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader)
            row = next(reader)

            assert headers == [
                "Record ID",
                "DOI",
                "Languages",
                "File Name",
                "Extracted Text",
                "Failed",
                "Supported",
            ]
            assert row == [
                "test123",
                "10.1234/test",
                "eng",
                "test.pdf",
                "Extracted text",
                "0",
                "1",
            ]

        # Verify methods were called correctly
        downloader.download_file.assert_called_once()
        mock_extractor.extract_file.assert_called_once()


def test_extract_documents_integration(test_config):
    """Integration test that makes real API calls and processes actual files."""
    # Skip if required environment variables are not set
    if not os.getenv("KCWORKS_API_KEY"):
        pytest.skip("KCWORKS_API_KEY environment variable not set")

    if not os.getenv("KCWORKS_API_URL"):
        pytest.skip("KCWORKS_API_URL environment variable not set")

    # Create downloader instance with test config
    downloader = KCWorksCorpusExtractor(config=test_config)

    # Run extraction with a small number of documents
    downloader.extract_documents(max_docs=1)

    # Verify output files were created
    assert test_config.EXTRACTED_TEXT_CSV_PATH.exists()

    # Verify CSV content
    with open(
        test_config.EXTRACTED_TEXT_CSV_PATH, "r", newline="", encoding="utf-8"
    ) as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        row = next(reader)

        # Verify headers
        assert headers == [
            "Record ID",
            "DOI",
            "Languages",
            "File Name",
            "Extracted Text",
            "Failed",
            "Supported",
        ]

        # Verify row content
        assert len(row) == 7  # All fields should be present
        assert row[0]  # Record ID should not be empty
        assert row[5] in ["0", "1"]  # Failed should be 0 or 1
        assert row[6] in ["0", "1"]  # Supported should be 0 or 1

    # Verify downloaded files directory contains files
    downloaded_files = list(Path(test_config.DOWNLOADED_FILES_PATH).glob("*"))
    assert len(downloaded_files) > 0


def test_document_extractor_pdf_chunking(tmp_path):
    """Test that PDF text extraction properly chunks the text."""
    # Create a mock PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF

    extractor = DocumentExtractor()

    with (
        patch("fitz.open") as mock_fitz,
        patch("kcworks_nlp_tools.kcworks_downloader.language.from_file") as mock_lang,
    ):
        # Mock the PDF content
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = (
            "This is a test sentence. This is another test sentence. " * 100
        )
        mock_doc.load_page.return_value = mock_page
        mock_doc.__len__.return_value = 1
        mock_fitz.return_value.__enter__.return_value = mock_doc
        mock_lang.return_value = "en"

        result = extractor.extract_file(str(pdf_path), "test.pdf")

        # Verify the result is a list of tuples
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

        # Verify each chunk is within the configured size limit
        for page_num, text, failed, supported in result:
            assert isinstance(page_num, int)
            assert isinstance(text, str)
            if text:  # If text extraction succeeded
                words = text.split()
                assert len(words) <= extractor.chunk_size
            assert isinstance(failed, bool)
            assert isinstance(supported, bool)


def test_document_extractor_tika_extraction(tmp_path):
    """Test that Tika-based extraction works for supported file types."""
    # Create a mock document file
    doc_path = tmp_path / "test.doc"
    doc_path.write_text("Test content")

    extractor = DocumentExtractor()

    with (
        patch("kcworks_nlp_tools.kcworks_downloader.parser.from_file") as mock_parser,
        patch("kcworks_nlp_tools.kcworks_downloader.language.from_file") as mock_lang,
    ):
        # Mock Tika's response
        mock_parser.return_value = {
            "content": "This is a test sentence. This is another test sentence."
        }
        mock_lang.return_value = "en"

        result = extractor.extract_file(str(doc_path), "test.doc")

        # Verify the result format
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 4 for item in result)

        # Check the first chunk
        page_num, text, failed, supported = result[0]
        assert page_num == 0  # Non-PDF files use page 0
        assert isinstance(text, str)
        assert not failed
        assert supported


def test_document_extractor_unsupported_file(tmp_path):
    """Test handling of unsupported file types."""
    # Create a mock file with unsupported extension
    file_path = tmp_path / "test.xyz"
    file_path.write_text("Test content")

    extractor = DocumentExtractor()
    result = extractor.extract_file(str(file_path), "test.xyz")

    # Verify the result indicates unsupported file type
    assert isinstance(result, list)
    assert len(result) == 1
    page_num, text, failed, supported = result[0]
    assert page_num == 0
    assert text is None
    assert failed
    assert not supported


def test_document_extractor_extraction_failure(tmp_path):
    """Test handling of extraction failures."""
    # Create a mock PDF file
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")  # Minimal valid PDF

    extractor = DocumentExtractor()

    with (
        patch("fitz.open", side_effect=Exception("Extraction failed")),
        patch("kcworks_nlp_tools.kcworks_downloader.language.from_file") as mock_lang,
    ):
        mock_lang.return_value = "en"
        result = extractor.extract_file(str(pdf_path), "test.pdf")

        # Verify the result indicates extraction failure
        assert isinstance(result, list)
        assert len(result) == 1
        page_num, text, failed, supported = result[0]
        assert page_num == 0
        assert text is None
        assert failed
        assert supported  # File type is supported even though extraction failed
