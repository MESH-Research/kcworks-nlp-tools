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
                },
                {
                    "id": "test456",
                    "pids": {"doi": {"identifier": "10.1234/test2"}},
                    "metadata": {"languages": [{"id": "eng"}, {"id": "spa"}]},
                    "files": {
                        "entries": {
                            "doc1.pdf": {},
                            "doc2.pdf": {},
                        }
                    },
                },
                {
                    "id": "test789",
                    "pids": {"doi": {"identifier": "10.1234/test3"}},
                    "metadata": {"languages": [{"id": "fra"}]},
                    "files": {
                        "entries": {
                            "presentation.pdf": {},
                            "notes.pdf": {},
                            "summary.pdf": {},
                        }
                    },
                },
            ]
        },
        "links": {"next": "http://test.com/next"},
    }
    return mock


@pytest.fixture
def mock_extractor():
    mock = Mock(spec=DocumentExtractor)

    # Return different text based on the file name
    def mock_extract_file(file_path, file_name):
        text_map = {
            "test.pdf": [("First document text", 1, False, True)],
            "doc1.pdf": [("Second document text", 1, False, True)],
            "doc2.pdf": [("Third document text", 1, False, True)],
            "presentation.pdf": [
                ("Fourth document text part 1", 1, False, True),
                ("Fourth document text part 2", 2, False, True),
            ],
            "notes.pdf": [(None, 1, True, True)],  # Extraction failed
            "summary.pdf": [("Sixth document text", 1, False, True)],
        }
        return text_map[file_name]

    mock.extract_file.side_effect = mock_extract_file
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

    # Set up the expected API URL
    expected_url = (
        f"{test_config.KCWORKS_API_URL}/{test_config.API_ENDPOINT}"
        f"?size={test_config.BATCH_SIZE}&page=1"
    )

    with (
        patch(
            "kcworks_nlp_tools.kcworks_downloader.DocumentExtractor",
            return_value=mock_extractor,
        ),
        patch("requests.get", return_value=mock_response) as mock_get,
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
        # Create downloader instance and patch its download_file method
        downloader = KCWorksCorpusExtractor(
            config=test_config, use_unique_filenames=False
        )

        # Mock download_file to fail for doc2.pdf
        def mock_download_file(file_name, record_id, local_filename, auth_headers):
            if file_name == "doc2.pdf":
                return None
            return str(downloaded_path / file_name)

        downloader.download_file = Mock(side_effect=mock_download_file)

        # Run extraction
        downloader.extract_documents(max_docs=6)  # Process all 6 files

        # Verify the API was called with the correct URL
        mock_get.assert_called_with(
            expected_url,
            headers={"Authorization": f"Bearer {test_config.KCWORKS_API_KEY}"},
        )

        # Verify CSV was created with correct content
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader)
            rows = list(reader)

            assert headers == [
                "Record ID",
                "DOI",
                "Languages",
                "File Name",
                "Extracted Text",
                "Failed",
                "Supported",
            ]

            # Verify all 7 files were processed (6 files + 1 extra chunk)
            assert len(rows) == 7

            # Verify first record
            assert rows[0] == [
                "test123",
                "10.1234/test",
                "eng",
                "test.pdf",
                "First document text",
                "0",
                "1",
            ]

            # Verify second record's files
            assert rows[1] == [
                "test456",
                "10.1234/test2",
                "eng,spa",
                "doc1.pdf",
                "Second document text",
                "0",
                "1",
            ]
            # Verify failed download is recorded correctly
            assert rows[2] == [
                "test456",
                "10.1234/test2",
                "eng,spa",
                "doc2.pdf",
                "[Download Failed]",
                "1",
                "1",
            ]

            # Verify third record's files
            assert rows[3] == [
                "test789",
                "10.1234/test3",
                "fra",
                "presentation.pdf",
                "Fourth document text part 1",
                "0",
                "1",
            ]
            # Verify second chunk of presentation.pdf
            assert rows[4] == [
                "test789",
                "10.1234/test3",
                "fra",
                "presentation.pdf",
                "Fourth document text part 2",
                "0",
                "1",
            ]
            # Verify failed extraction is recorded correctly
            assert rows[5] == [
                "test789",
                "10.1234/test3",
                "fra",
                "notes.pdf",
                "[Error: Extraction Failed]",
                "1",
                "1",
            ]
            assert rows[6] == [
                "test789",
                "10.1234/test3",
                "fra",
                "summary.pdf",
                "Sixth document text",
                "0",
                "1",
            ]

        # Verify methods were called correctly
        assert downloader.download_file.call_count == 6
        # extract_file should only be called 5 times since one download failed
        assert mock_extractor.extract_file.call_count == 5

        # Verify first file download
        downloader.download_file.assert_any_call(
            "test.pdf",
            "test123",
            str(downloaded_path / "test.pdf"),
            {"Authorization": f"Bearer {test_config.KCWORKS_API_KEY}"},
        )

        # Verify failed file download
        downloader.download_file.assert_any_call(
            "doc2.pdf",
            "test456",
            str(downloaded_path / "doc2.pdf"),
            {"Authorization": f"Bearer {test_config.KCWORKS_API_KEY}"},
        )

        # Verify first file extraction
        mock_extractor.extract_file.assert_any_call(
            str(downloaded_path / "test.pdf"), "test.pdf"
        )


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


def test_document_extractor_pdf_chunking(test_config):
    """Test that PDF text extraction properly chunks the text."""
    # Use the sample PDF file
    pdf_path = Path("tests/sample_files/wind_in_the_willows.pdf")
    assert pdf_path.exists(), "Sample PDF file not found"

    extractor = DocumentExtractor(config=test_config)

    with patch("kcworks_nlp_tools.kcworks_downloader.language.from_file") as mock_lang:
        mock_lang.return_value = "en"

        result = extractor.extract_file(str(pdf_path), "wind_in_the_willows.pdf")
        print(result)

        # Verify the result is a list of tuples
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

        # Verify each chunk is within the configured size limit
        for text, page_num, failed, supported in result:
            assert isinstance(page_num, int)
            assert isinstance(text, str)
            if text:  # If text extraction succeeded
                words = text.split()
                assert len(words) <= extractor.chunk_size
            assert isinstance(failed, bool)
            assert isinstance(supported, bool)

        # Verify chunk properties
        assert len(result) > 1, "Expected multiple chunks for sample.pdf"

        # First chunk should be from page 1
        first_text, first_page, first_failed, first_supported = result[0]
        assert first_page == 1, "First chunk should be from page 1"
        assert not first_failed, "First chunk should not be marked as failed"
        assert first_supported, "First chunk should be marked as supported"
        assert first_text, "First chunk should have text content"

        # Last chunk should have the same flags
        last_text, last_page, last_failed, last_supported = result[-1]
        assert not last_failed, "Last chunk should not be marked as failed"
        assert last_supported, "Last chunk should be marked as supported"
        assert last_text, "Last chunk should have text content"

        # Verify chunks are in page order
        for i in range(len(result) - 1):
            current_page = result[i][1]
            next_page = result[i + 1][1]
            assert current_page <= next_page, "Chunks should be in page order"


def test_document_extractor_tika_extraction(test_config):
    """Test that Tika-based extraction works for supported file types."""
    # Use the sample DOC file
    doc_path = Path("tests/sample_files/sample.doc")
    assert doc_path.exists(), "Sample DOC file not found"

    extractor = DocumentExtractor(config=test_config)

    with patch("kcworks_nlp_tools.kcworks_downloader.language.from_file") as mock_lang:
        mock_lang.return_value = "en"

        result = extractor.extract_file(str(doc_path), "sample.doc")
        print(result)  # Print results for debugging

        # Verify the result is a list of tuples
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)

        # Verify each chunk is within the configured size limit
        for text, page_num, failed, supported in result:
            assert isinstance(page_num, int)
            assert isinstance(text, str)
            if text:  # If text extraction succeeded
                words = text.split()
                assert len(words) <= extractor.chunk_size
            assert isinstance(failed, bool)
            assert isinstance(supported, bool)

        # Verify chunk properties
        assert len(result) > 1, "Expected multiple chunks for sample.doc"

        # First chunk should have content
        first_text, first_page, first_failed, first_supported = result[0]
        assert first_page == 0, "Non-PDF files should use page 0"
        assert not first_failed, "First chunk should not be marked as failed"
        assert first_supported, "First chunk should be marked as supported"
        assert first_text, "First chunk should have text content"

        # Last chunk should have the same flags
        last_text, last_page, last_failed, last_supported = result[-1]
        assert last_page == 0, "Non-PDF files should use page 0"
        assert not last_failed, "Last chunk should not be marked as failed"
        assert last_supported, "Last chunk should be marked as supported"
        assert last_text, "Last chunk should have text content"

        # Verify all chunks use page 0 for non-PDF files
        for text, page_num, failed, supported in result:
            assert page_num == 0, "Non-PDF files should use page 0"


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
