from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    result_img = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Predict
            results = model.predict(source=filepath, save=True, project=UPLOAD_FOLDER, name="predict", exist_ok=True)

            result_img = os.path.join(UPLOAD_FOLDER, "predict", os.path.basename(filepath))

    return render_template("index.html", result_img=result_img)

if __name__ == "__main__":
    app.run(debug=True)
    

