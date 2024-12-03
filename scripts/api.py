from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from detect_mask import detect_mask
import cv2
import numpy as np

app = Flask(__name__)
model = load_model("mask_detector.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    label = detect_mask(img, model)
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=True)
