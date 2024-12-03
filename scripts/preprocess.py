import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    images, labels = [], []
    categories = ["with_mask", "without_mask"]
    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    images = np.array(images) / 255.0
    labels = to_categorical(labels, num_classes=2)
    return images, labels

# Example usage
# X, y = load_data("dataset")
