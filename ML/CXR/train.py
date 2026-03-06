import torch
import torch.nn as nn
import torch.optim as optim

from model import PneumoniaCNN
from dataloader import train_loader, val_loader


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# model
model = PneumoniaCNN().to(device)


# loss function
criterion = nn.CrossEntropyLoss()


# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# epochs
epochs = 2


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


# save model
torch.save(model.state_dict(),"pneumonia_cnn.pth")

print("Model saved!")