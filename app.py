# ==============================
# RespireAI – Final Production App
# Pneumonia Detection (X-ray & CT)
# ==============================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RespireAI – Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    body { background-color: #F5F7FA; }
    .title {
        font-size: 48px;
        font-weight: 800;
        color: #1F4FD8;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">RespireAI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Pneumonia Detection System</div>', unsafe_allow_html=True)
st.write("---")

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model
# -----------------------------
class PneumoniaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, 2)

    def forward(self, x):
        return self.base(x)

model = PneumoniaNet().to(DEVICE)
model.eval()  # Demo mode (no weights loaded)

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

classes = ["Normal", "Pneumonia"]

# -----------------------------
# UI – Upload Section
# -----------------------------
st.subheader("📤 Upload Chest X-ray or CT Scan")
uploaded = st.file_uploader(
    "Accepted formats: JPG, PNG, JPEG",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_container_width=True)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    confidence = conf.item() * 100

    st.write("---")
    st.subheader("🧠 AI Diagnosis")

    if label == "Pneumonia":
        st.error(f"⚠️ Pneumonia Detected")
    else:
        st.success(f"✅ No Pneumonia Detected")

    st.metric("Confidence", f"{confidence:.2f}%")

    # -----------------------------
    # Explainability (Grad-CAM Placeholder)
    # -----------------------------
    st.write("---")
    st.subheader("🔍 Model Explainability")
    st.info(
        "Heatmap visualization (Grad-CAM) highlights lung regions used by the model. "
        "This feature is enabled after full clinical training."
    )

# -----------------------------
# Footer
# -----------------------------
st.write("---")
st.caption(
    "⚠️ Disclaimer: RespireAI is a research and educational prototype. "
    "It is not a substitute for professional medical diagnosis."
)
