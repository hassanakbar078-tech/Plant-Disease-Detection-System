"""
download_dataset.py — Helper script to download the PlantVillage dataset from Kaggle.

Pre-requisites:
    1. Install the Kaggle CLI:      pip install kaggle
    2. Create a Kaggle API token:
       - Go to https://www.kaggle.com/settings → "API" → "Create New Token"
       - This downloads kaggle.json.  Place it at ~/.kaggle/kaggle.json
         (Linux/Mac) or %USERPROFILE%\.kaggle\kaggle.json (Windows).
    3. Run this script:             python download_dataset.py

The dataset will be extracted to ./PlantVillage/
"""

import os
import zipfile
import subprocess
import sys


DATASET_SLUG = "arjuntejaswi/plant-village"
DOWNLOAD_DIR = "."


def main():
    try:
        import kaggle  # noqa: F401 — triggers credential check
    except ImportError:
        print("Kaggle package not found. Installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    print(f"Downloading dataset: {DATASET_SLUG} …")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            DOWNLOAD_DIR,
        ]
    )

    zip_path = os.path.join(DOWNLOAD_DIR, "plant-village.zip")
    if not os.path.exists(zip_path):
        # Kaggle sometimes names the file differently
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith(".zip"):
                zip_path = os.path.join(DOWNLOAD_DIR, f)
                break

    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DOWNLOAD_DIR)

    print("\n✅  Dataset ready.")
    print("    The 'PlantVillage' folder (containing class sub-folders) should now exist.")
    print("    Run training with:")
    print("        python train.py --data_dir PlantVillage\n")


if __name__ == "__main__":
    main()
