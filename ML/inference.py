import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------- CONFIG ----------
MODEL_PATH = "ML/resnet18_drug_classifier.pth"

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ---------- TRANSFORM ----------
def ensure_rgb(img):
    return img.convert("RGB")

transform = transforms.Compose([
    transforms.Lambda(ensure_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================

def predict_drug_from_image(image, threshold=0.35):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    confidence = confidence.item()


    return class_names[pred_idx.item()], confidence



# ---------- TEST  ----------
if __name__ == "__main__":
    test_image = "ML/images (1).jpg"
    pred = predict_drug_from_image(test_image)
    print("Predicted drug:", pred)