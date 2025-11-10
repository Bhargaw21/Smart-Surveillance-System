from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # enables cross-origin requests

def base64_to_image(base64_str):
    """
    Convert base64 string to OpenCV image
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    decoded = base64.b64decode(base64_str)
    np_arr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()
        img1_str = data.get("img1")
        img2_str = data.get("img2")
        model_name = data.get("model_name", "Facenet")
        detector_backend = data.get("detector_backend", "opencv")
        distance_metric = data.get("distance_metric", "cosine")
        enforce_detection = data.get("enforce_detection", False)

        if not img1_str or not img2_str:
            return jsonify({"error": "img1 or img2 missing"}), 400

        img1 = base64_to_image(img1_str)
        img2 = base64_to_image(img2_str)

        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
