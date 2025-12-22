import torch
import cv2
import numpy as np
import pickle
from torchvision import transforms
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load FULL model 
model = torch.load("model/detector.pth", map_location="cpu", weights_only=False)
model.eval()

# Load label encoder
with open("model/le.pickle", "rb") as f:
    le = pickle.load(f)

def predict_from_bytes(image_bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_tensor).unsqueeze(0)

    # Inference
    with torch.no_grad():
        _, label_logits = model(img_tensor)  # Ignore bbox output
        probs = torch.softmax(label_logits, dim=1)
        idx = probs.argmax().item()
        label = le.inverse_transform([idx])[0]
        confidence = probs[0, idx].item()

    return label, confidence