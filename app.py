from flask import Flask, render_template, request
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime

app = Flask(__name__)

# Folder untuk upload & hasil
UPLOAD_FOLDER = "static/uploads"
PREDICT_FOLDER = os.path.join(UPLOAD_FOLDER, "predict")

os.makedirs(PREDICT_FOLDER, exist_ok=True)

# Load model YOLOv11
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model '{model_path}' tidak ditemukan.")

model = YOLO(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            # Simpan file upload
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Deteksi dengan YOLOv11
            results = model.predict(source=filepath, save=True, project=UPLOAD_FOLDER, name="predict", exist_ok=True)

            # Path hasil deteksi (gambar dengan bounding box)
            result_img = os.path.join("static", "uploads", "predict", os.path.basename(filepath))

    return render_template("index.html", result_img=result_img)

# Tidak pakai app.run() karena Render pakai Gunicorn
# Gunicorn akan mencari variabel `app` di file ini
