import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_mask(frame, model):
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return "No Mask" if pred[0] > pred[1] else "Mask"

if __name__ == "__main__":
    model = load_model("./models/mask_detector.h5")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label = detect_mask(frame, model)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Mask Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
