# predict.py
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/cnn-parameters-improvement-23-0.91.model"  # repo model filename
IMAGE_SIZE = (240, 240)  # same shape used in training (from README)

def load_model():
    # If load fails, see troubleshooting below.
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    # The repo says they crop brain region; if you don't crop, use center crop or whole image:
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    # model expects shape (1, 240, 240, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image_path):
    model = load_model()
    x = preprocess_image(image_path)
    pred = model.predict(x)  # output probably a single sigmoid value
    # If model outputs probability (sigmoid): value close to 1 => tumor (yes)
    prob = float(pred.ravel()[0])
    label = "tumor" if prob >= 0.5 else "no_tumor"
    return {"probability": prob, "label": label}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)
    img_path = sys.argv[1]
    res = predict(img_path)
    print(res)
