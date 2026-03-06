#3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from .dataloader import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
from torchvision.models import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_classes = 3
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in model.parameters():
    param.requires_grad = False


for param in model.fc.parameters():
    param.requires_grad = True


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

model = model.to(device)


num_epochs = 10

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    # ========= TRAIN =========
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    # ========= VALIDATION =========
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2%}")


    ### testing

from dataloader import test_loader

print("\nEvaluating on TEST set")
print("-" * 30)

model.eval()

correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.2%}")



#confusion amtrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class_names = ["acyclovir", "ampicillin", "vancomycin"]

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


import os

SAVE_PATH = "ML/resnet18_drug_classifier.pth"
os.makedirs("ML", exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, SAVE_PATH)

print(f"\nModel saved to {SAVE_PATH}")