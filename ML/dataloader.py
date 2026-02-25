#2
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


def ensure_rgb(img):
    return img.convert("RGB")
# augmentation
train_transform = transforms.Compose([
    transforms.Lambda(ensure_rgb),
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Lambda(ensure_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


## labeling
DATASET_DIR = "ML/dataset"

train_dataset = ImageFolder(
    root=os.path.join(DATASET_DIR, "train"),
    transform=train_transform
)

val_dataset = ImageFolder(
    root=os.path.join(DATASET_DIR, "val"),
    transform=val_test_transform
)

test_dataset = ImageFolder(
    root=os.path.join(DATASET_DIR, "test"),
    transform=val_test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":
    print("Classes:", train_dataset.classes)

    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch labels:", labels)