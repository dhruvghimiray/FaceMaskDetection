from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import load_data
import os

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    data_dir = "dataset"
    X, y = load_data(data_dir)
    model = build_model()
    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.2)
    # model_dir = "./models"
    # os.makedirs(model_dir, exist_ok=True)
    model.save("models/mask_detector.h5")