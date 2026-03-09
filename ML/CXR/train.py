import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from model import PneumoniaCNN
from dataloader import train_loader, val_loader, train_dataset

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# model
model = PneumoniaCNN().to(device)


# loss function
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets
)

class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0003)


# epochs
epochs = 10


for epoch in range(epochs):

    # training mode
    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_loss = running_loss / len(train_loader)


    # validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    val_accuracy = correct / total


    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print("---------------------------")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(confusion_matrix(all_labels, all_preds))
# save model
torch.save(model.state_dict(),"pneumonia_cnn.pth")

print("Model saved!")