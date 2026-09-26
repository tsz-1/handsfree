"""Download the MediaPipe hand landmarker model into models/."""

import urllib.request
from pathlib import Path

from handsfree.tracker import MODEL_URL

DEST = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def main():
    if DEST.exists():
        print(f"Model already exists at {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, DEST)
    print(f"Saved to {DEST}")


if __name__ == "__main__":
    main()
