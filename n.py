from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# تحميل نموذج DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2 = dinov2.to(device)

# تحميل SVM والمحول
clf = joblib.load("models/plant_identifier_SVM_model.pkl")
le = joblib.load("models/plant_label_encoder.pkl")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_embedding(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = dinov2(tensor).squeeze().cpu().numpy().reshape(1, -1)
    return emb

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        emb = extract_embedding(img)
        pred = clf.predict(emb)[0]
        label = le.inverse_transform([pred])[0]

        return JSONResponse({"class": label})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
