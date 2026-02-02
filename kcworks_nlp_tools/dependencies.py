import tarfile
import os
from pathlib import Path

import requests
from pydantic import BaseModel

from kcworks_nlp_tools.util import get_package_root


def download_tika_binary():
    """Download the precompiled Tika binary and save it in the lib folder"""
    local_file_path = Path(__file__).parent / "lib" / "tika-server-standard-3.1.0.jar"
    if not local_file_path.exists():
        tika_response = requests.get(
            "https://dlcdn.apache.org/tika/3.1.0/tika-server-standard-3.1.0.jar"  # noqa: E501
        )
        with open(local_file_path, mode="wb") as local_file:
            local_file.write(tika_response.content)
    else:
        print(f"Tika binary already exists in {local_file_path}")


class ExtractionResult(BaseModel):
    unextracted_files: list[str] = []
    missing_files: list[str] = []
    extracted_files: list[str] = []


def extract_fast_files(paths: list[str]) -> None:
    """Extract the fast vocabulary JSON files for use.

    Check whether each source file exists already. If it doesn't,
    look for a tar.bz2 archive to extract. If no tar archive
    exists, report the missing file.

    Raises:
        RuntimeError: if no source files exist or can be extracted
            for the provided paths.
    """
    result = ExtractionResult()

    dir = get_package_root()
    for path in paths:
        source_path = Path(dir) / path
        tar_path = Path(dir) / path.replace("jsonl", "tar.bz2")

        if os.path.exists(source_path):
            result.extracted_files.append(path)
        elif os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:bz2") as tar:
                try:
                    assert len(tar.getnames()) == 1
                    subjects_dir = tar_path.parent
                    tar.extractall(subjects_dir)
                    assert os.path.exists(source_path)
                except AssertionError:
                    raise ValueError(
                        "Archive did not contain the expected files or could not be extracted"
                    )
        else:
            result.missing_files.append(path)

    if len(result.extracted_files) == 0:
        raise RuntimeError

    return result
