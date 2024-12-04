import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the pretrained model locally (MobileNetV2 without top layers)
model = load_model("models/mobilenet_v2_face_detector.h5")

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function for feature extraction using MobileNetV2
def extract_features(face, model):
    img = cv2.resize(face, (128, 128)) / 255.0  # Resize to fit model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    features = model.predict(img)
    return features

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe Face Detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (invert the webcam output)
        frame = cv2.flip(frame, 1)  # Change 1 to 0 for vertical flip or -1 for both

        # Convert the frame to RGB (MediaPipe uses RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the face region
                face = frame[y:y + h, x:x + w]
                
                if face.size > 0:
                    # Extract features from the face using MobileNetV2
                    features = extract_features(face, model)

                    # You can use features for further processing, but for now, let's display the bounding box
                    cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display the output frame
        cv2.imshow("Face Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
