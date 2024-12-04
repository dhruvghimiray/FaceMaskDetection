from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import save_model

# Load MobileNetV2 with pretrained weights from ImageNet
model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Save the model locally
save_model(model, "models/mobilenet_v2_face_detector.h5")
print("Model saved locally as mobilenet_v2_face_detector.h5")
