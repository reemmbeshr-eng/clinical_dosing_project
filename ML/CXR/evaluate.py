import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from model import PneumoniaCNN
from dataloader import test_loader


# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD MODEL
# -----------------------------

model = PneumoniaCNN().to(device)

model.load_state_dict(
    torch.load("pneumonia_cnn.pth", map_location=device)
)

model.eval()


# -----------------------------
# STORAGE
# -----------------------------

all_preds = []
all_labels = []
all_probs = []


# -----------------------------
# EVALUATION LOOP
# -----------------------------

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:,1].cpu().numpy())


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# -----------------------------
# METRICS
# -----------------------------

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)


print("\n===== MODEL PERFORMANCE =====\n")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC      :", auc)


# -----------------------------
# SAVE METRICS (FOR DASHBOARD)
# -----------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auc": float(auc)
}

with open("model_metrics.json","w") as f:
    json.dump(metrics,f)


# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix")
print(cm)

np.save("confusion_matrix.npy", cm)


# -----------------------------
# CONFUSION MATRIX PLOT
# -----------------------------

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Pneumonia"],
    yticklabels=["Normal","Pneumonia"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# -----------------------------
# ROC CURVE
# -----------------------------

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

roc_data = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist()
}

with open("roc_data.json","w") as f:
    json.dump(roc_data,f)


plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()