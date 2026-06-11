"""
Download emotion model from alternative sources
"""
import urllib.request
import os
import sys

# Try multiple sources
MODELS = [
    {
        "name": "FER2013 (Kaggle)",
        "url": "https://github.com/Gr33nW33d/Emotion-Recognition/raw/main/models/emotion_model.hdf5",
        "size": "~40 MB"
    },
    {
        "name": "Emotion Detection (Alternative)",
        "url": "https://github.com/muxspace/facial_expressions/raw/master/model.hdf5",
        "size": "~20 MB"
    }
]

def download(url, save_path):
    """Download file with progress bar"""
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = int(pct / 2)
            sys.stdout.write(f"\r[{'█'*bar}{'░'*(50-bar)}] {pct:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, save_path, reporthook=progress)
        print(f"\n✅ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return False

if __name__ == "__main__":
    save_path = "emotion_model.hdf5"

    if os.path.exists(save_path):
        print(f"✅ Model already exists: {save_path}")
        sys.exit(0)

    print("=" * 60)
    print("  Alternative Model Sources")
    print("=" * 60)

    for i, model in enumerate(MODELS, 1):
        print(f"\n{i}. {model['name']} (~{model['size']})")
        print(f"   Trying: {model['url']}")

        if download(model['url'], save_path):
            print(f"\n✅ Saved → {save_path}")
            break
    else:
        print("\n❌ All sources failed. Download manually:")
        print("\n1. Go to: https://github.com/oarriaga/face_classification/releases")
        print("2. Download: fer2013_mini_XCEPTION.102-0.66.hdf5")
        print(f"3. Save to: {os.path.abspath(save_path)}")
