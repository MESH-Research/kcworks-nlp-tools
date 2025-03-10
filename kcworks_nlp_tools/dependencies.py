import requests
from pathlib import Path


def download_tika_binary():
    """Download the precompiled Tika binary and save it in the lib folder
    """
    local_file_path = (
        Path(__file__).parent / "lib" / "tika-server-standard-3.1.0.jar"
    )
    if not local_file_path.exists():
        tika_response = requests.get(
            "https://dlcdn.apache.org/tika/3.1.0/tika-server-standard-3.1.0.jar"  # noqa: E501
        )
        with open(local_file_path, mode="wb") as local_file:
            local_file.write(tika_response.content)
    else:
        print(f"Tika binary already exists in {local_file_path}")

