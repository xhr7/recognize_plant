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
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل نموذج DINOv2 إذا ما كان موجود
MODEL_PATH = "dinov2_model.pt"
FILE_ID = "1gD6k2kwNAB-mKgwSkRb8wOzs2NF0xD8w"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading DINOv2 model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)

# تحميل النموذج
dinov2 = torch.jit.load(MODEL_PATH).eval()
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
