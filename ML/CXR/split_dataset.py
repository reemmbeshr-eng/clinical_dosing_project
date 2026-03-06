import os
import random
import shutil

dataset_path = r"D:/AI diploma/clinical_dosing_project/ML/CXR/CXR_dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = ["NORMAL", "PNEUMONIA"]

for c in classes:

    class_path = os.path.join(dataset_path, c)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in zip(
        ["train","val","test"],
        [train_images, val_images, test_images]
    ):

        split_folder = os.path.join(dataset_path, split, c)
        os.makedirs(split_folder, exist_ok=True)

        for img in split_images:

            src = os.path.join(class_path, img)
            dst = os.path.join(split_folder, img)

            shutil.copy(src, dst)

print("Dataset split completed")