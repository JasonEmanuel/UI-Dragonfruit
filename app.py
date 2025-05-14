import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

# Custom CSS style
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

# Load model
model = YOLO("best.pt")

# Title and description
st.title("🍉 Deteksi Kematangan Buah Naga")
st.write("Upload gambar buah naga, lalu sistem akan memprediksi tingkat kematangannya menggunakan model YOLOv11.")

# File uploader
uploaded_file = st.file_uploader("Unggah Gambar Buah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file
    uploads_dir = "static/uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uploaded_file.name
    filepath = os.path.join(uploads_dir, filename)

    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar yang diupload
    st.image(Image.open(filepath), caption="📷 Gambar yang Diunggah", use_column_width=True)

    # Deteksi
    with st.spinner("🔍 Melakukan deteksi..."):
        results = model.predict(source=filepath, save=True, project=uploads_dir, name="predict", exist_ok=True)
        result_img_path = os.path.join(uploads_dir, "predict", os.path.basename(filepath))

    # Tampilkan hasil
    if os.path.exists(result_img_path):
        st.success("✅ Hasil Deteksi:")
        st.image(result_img_path, use_column_width=True)
    else:
        st.warning("⚠️ Tidak ada objek terdeteksi.")
