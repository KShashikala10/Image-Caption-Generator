# scripts/extract_features.py
import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Paths
DATA_DIR = r"C:\Users\Admin\Desktop\Image_Caption_Generator\data\Flickr8k_Dataset"
OUT_FILE = os.path.join("..", "features", "image_features.pkl")

def build_model():
    base_model = InceptionV3(weights="imagenet")
    model = Model(base_model.input, base_model.layers[-2].output)
    return model

def extract_features(model, img_dir):
    features = {}
    filenames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    print(f"🖼️ Found {len(filenames)} images in {img_dir}")

    for img_name in tqdm(filenames):
        img_path = os.path.join(img_dir, img_name)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        feat = model.predict(x, verbose=0)

        # ✅ FIX: remove .jpg extension to match caption IDs
        key = os.path.splitext(img_name)[0]
        features[key] = feat.flatten()

    return features

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    print("🚀 Building model and extracting features...")
    model = build_model()
    feats = extract_features(model, DATA_DIR)
    with open(OUT_FILE, "wb") as f:
        pickle.dump(feats, f)
    print(f"✅ Saved features to: {OUT_FILE}")
