from pathlib import Path

import requests


DATA_URL = "https://snap.stanford.edu/data/email-Enron.txt.gz"
TARGET_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "email-Enron.txt.gz"


def main() -> None:
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    TARGET_PATH.write_bytes(response.content)
    print(f"Saved dataset to: {TARGET_PATH}")


if __name__ == "__main__":
    main()
