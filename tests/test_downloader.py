import pytest
import csv
import os
from unittest.mock import Mock, patch
from pathlib import Path
from kcworks_nlp_tools.kcworks_downloader import KCWorksCorpusExtractor, DocumentExtractor
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
                    "files": {
                        "entries": {
                            "test.pdf": {}
                        }
                    }
                }
            ]
        },
        "links": {}
    }
    return mock


@pytest.fixture
def mock_extractor():
    mock = Mock(spec=DocumentExtractor)
    mock.extract_file.return_value = ("Extracted text", False, True)
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

    with patch("kcworks_nlp_tools.kcworks_downloader.DocumentExtractor", return_value=mock_extractor), \
         patch("requests.get", return_value=mock_response), \
         patch("kcworks_nlp_tools.kcworks_downloader.config") as mock_config, \
         patch("kcworks_nlp_tools.kcworks_downloader.os.remove"):

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
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)
            row = next(reader)

            assert headers == [
                "Record ID", "DOI", "Languages", "File Name", "Page Number",
                "Extracted Text", "Failed", "Supported"
            ]
            assert row == [
                "test123", "10.1234/test", "eng", "test.pdf", "0",
                "Extracted text", "0", "1"
            ]

        # Verify methods were called correctly
        downloader.download_file.assert_called_once()
        mock_extractor.extract_file.assert_called_once()


@pytest.mark.integration
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
    with open(test_config.EXTRACTED_TEXT_CSV_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        headers = next(reader)
        row = next(reader)

        # Verify headers
        assert headers == [
            "Record ID", "DOI", "Languages", "File Name", "Page Number",
            "Extracted Text", "Failed", "Supported"
        ]

        # Verify row content
        assert len(row) == 7  # All fields should be present
        assert row[0]  # Record ID should not be empty
        assert row[5] in ["0", "1"]  # Failed should be 0 or 1
        assert row[6] in ["0", "1"]  # Supported should be 0 or 1

    # Verify downloaded files directory contains files
    downloaded_files = list(Path(test_config.DOWNLOADED_FILES_PATH).glob("*"))
    assert len(downloaded_files) > 0

