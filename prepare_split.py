import os
import shutil
import random
from pathlib import Path

# Base data directory
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

# Train/validation split ratio
SPLIT_RATIO = 0.8  

def prepare_split():
    # Create train/val folders
    for folder in [TRAIN_DIR, VAL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    # Loop over each class folder inside raw/
    for class_name in os.listdir(RAW_DIR):
        class_path = RAW_DIR / class_name
        if not class_path.is_dir():
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Create class subfolders inside train/val
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(class_path / img, TRAIN_DIR / class_name / img)

        for img in val_images:
            shutil.copy(class_path / img, VAL_DIR / class_name / img)

    print(f"✅ Split done: {TRAIN_DIR} and {VAL_DIR} created.")

if __name__ == "__main__":
    prepare_split()
