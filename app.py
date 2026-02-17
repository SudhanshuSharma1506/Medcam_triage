from flask import Flask, request, jsonify, render_template

import os
import numpy as np
import cv2
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained SVM model
try:
    model = joblib.load("svm_model_poly.pkl")
    print("SVM model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")




@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # -----------------------------
        # 1️⃣ Read original image
        # -----------------------------
        original_img = cv2.imread(filepath)

        # Get original file size
        original_size = os.path.getsize(filepath)

        # -----------------------------
        # 2️⃣ Compress image
        # -----------------------------
        compressed_img = cv2.resize(original_img, (64, 64))

        compressed_path = os.path.join(
            UPLOAD_FOLDER, "compressed_" + file.filename
        )

        cv2.imwrite(
            compressed_path,
            compressed_img,
            [cv2.IMWRITE_JPEG_QUALITY, 40]
        )

        compressed_size = os.path.getsize(compressed_path)

        compression_ratio = round(original_size / compressed_size, 2)

        img = compressed_img

        # -----------------------------
        # 3️⃣ Feature Extraction (11 features)
        # -----------------------------
        b, g, r = cv2.split(img)

        features = [
            np.mean(r),
            np.mean(g),
            np.mean(b),
            np.std(r),
            np.std(g),
            np.std(b),
            img.shape[0],                     # height
            img.shape[1],                     # width
            img.shape[0] * img.shape[1],      # area
            img.shape[1] / img.shape[0],      # aspect ratio
            np.mean(img)                      # overall brightness
        ]

        features = np.array(features).reshape(1, -1)

        # -----------------------------
        # 4️⃣ Prediction
        # -----------------------------
        prediction = model.predict(features)[0]

        if str(prediction).lower() == "malignant":
            result = "HIGH RISK"
        else:
            result = "LOW RISK"

        # -----------------------------
        # 5️⃣ Return Response
        # -----------------------------
        return jsonify({
            "risk": result,
            "raw_prediction": str(prediction),
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
