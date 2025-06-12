import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
from datetime import datetime

# Load YOLO model
model = YOLO("best.pt")

# Custom CSS (optional)
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #f0fdf4, #dcfce7);
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background-color: #16a34a;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #15803d;
    }
    .stFileUploader {
        background-color: #f0fdf4;
        padding: 1em;
        border: 2px dashed #4ade80;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Instructions
st.title("🍉 Deteksi Kematangan Buah Naga")
st.write("Upload gambar buah naga, dan sistem akan mendeteksi kematangannya menggunakan YOLOv11.")

# Function: Auto-Orient
def correct_image_orientation(image):
    return ImageOps.exif_transpose(image)

# Function: Resize to 640x640
def resize_image(image, size=(640, 640)):
    return image.resize(size)

# Function: Adaptive Contrast Enhancement (Equalization)
def equalize_image(pil_img):
    img_np = np.array(pil_img)
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

# File uploader
uploaded_file = st.file_uploader("Unggah gambar (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = correct_image_orientation(img)
    img = resize_image(img)
    img = equalize_image(img)

    # Tampilkan gambar hasil preprocessing
    st.image(img, caption="📷 Gambar Setelah Preprocessing", use_container_width=True)

    # Deteksi dengan YOLOv11
    with st.spinner("🔍 Mendeteksi buah naga..."):
        results = model.predict(img, conf=0.25, save=False)

        # Tampilkan hasil
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="✅ Hasil Deteksi", channels="BGR", use_container_width=True)
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi.")
