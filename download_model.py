"""
download_model.py
==================
Run this ONCE to download the correct pre-trained emotion model.
It downloads a FULL model (architecture + weights bundled together).

Usage:
    python download_model.py
"""

import urllib.request
import os
import sys


MODEL_URL  = "https://github.com/oarriaga/face_classification/releases/download/v0.1/fer2013_mini_XCEPTION.102-0.66.hdf5"
SAVE_PATH  = "emotion_model.hdf5"


def download():
    if os.path.exists(SAVE_PATH):
        size = os.path.getsize(SAVE_PATH) / 1024 / 1024
        print(f"Model already exists: {SAVE_PATH} ({size:.1f} MB)")
        return True

    print("=" * 60)
    print("  Downloading pre-trained Emotion Detection Model")
    print("  Source : oarriaga/face_classification (FER-2013)")
    print("  Size   : ~2 MB")
    print("=" * 60)

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct / 2)
            sys.stdout.write(f"\r  [{'█'*bar}{'░'*(50-bar)}] {pct:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, SAVE_PATH, reporthook=progress)
        print(f"\n\nSaved → {SAVE_PATH}")
        size = os.path.getsize(SAVE_PATH) / 1024 / 1024
        print(f"Size  : {size:.1f} MB")
        print("\nNow restart your server:")
        print("  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return True
    except Exception as e:
        print(f"\n\nDownload failed: {e}")
        print("\nManual download:")
        print(f"  1. Open this URL in your browser:")
        print(f"     {MODEL_URL}")
        print(f"  2. Save the file as: {os.path.abspath(SAVE_PATH)}")
        return False


if __name__ == "__main__":
    download()