#1

import os
import shutil
import random

# =============================
# Configuration
# =============================
RANDOM_SEED = 42

RAW_DIR = "ML/rawimages"
OUTPUT_DIR = "ML/dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

random.seed(RANDOM_SEED)


# =============================
# Helper functions
# =============================
def is_image(file_name):
    return file_name.lower().endswith(IMG_EXTENSIONS)


def split_class(class_name):
    class_src = os.path.join(RAW_DIR, class_name)

    images = [
        f for f in os.listdir(class_src)
        if is_image(f)
    ]

    if len(images) == 0:
        print(f"[WARNING] No images found for class: {class_name}")
        return

    random.shuffle(images)

    n_total = len(images)
    n_train = int(SPLITS["train"] * n_total)
    n_val = int(SPLITS["val"] * n_total)

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, files in split_map.items():
        dst_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        for file_name in files:
            src_path = os.path.join(class_src, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            shutil.copy(src_path, dst_path)

    print(
        f"[OK] {class_name}: "
        f"{len(split_map['train'])} train | "
        f"{len(split_map['val'])} val | "
        f"{len(split_map['test'])} test"
    )


# =============================
# Main execution
# =============================
def main():
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Raw images folder not found: {RAW_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    classes = [
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
    ]

    if len(classes) == 0:
        raise RuntimeError("No class folders found in raw_images")

    print(f"Found classes: {classes}")

    for cls in classes:
        split_class(cls)

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()