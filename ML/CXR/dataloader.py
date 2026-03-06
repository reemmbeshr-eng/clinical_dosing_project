import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from preprocessing import train_transform, val_test_transform

# مسار الفولدر الحالي (ML/CXR)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# مسار الداتا
DATASET_PATH = os.path.join(BASE_DIR, "CXR_dataset")

train_dataset = ImageFolder(
    os.path.join(DATASET_PATH, "train"),
    transform=train_transform
)

val_dataset = ImageFolder(
    os.path.join(DATASET_PATH, "val"),
    transform=val_test_transform
)

test_dataset = ImageFolder(
    os.path.join(DATASET_PATH, "test"),
    transform=val_test_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))