# scripts/inference.py
import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "models", "caption_model.h5")
TOKENIZER_FILE = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
SEQS_FILE = os.path.join(BASE_DIR, "models", "sequences.npz")

# -------------------------------------------------
# Load tokenizer and model
# -------------------------------------------------
def load_tokenizer_and_model():
    print("📦 Loading model and tokenizer...")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_FILE}")
    if not os.path.exists(TOKENIZER_FILE):
        raise FileNotFoundError(f"❌ Tokenizer not found at {TOKENIZER_FILE}")

    model = load_model(MODEL_FILE)
    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    meta = np.load(SEQS_FILE, allow_pickle=True)
    max_length = int(meta["max_len"])

    print("✅ Model and tokenizer loaded successfully.")
    return model, tokenizer, max_length

# -------------------------------------------------
# Feature extraction (InceptionV3)
# -------------------------------------------------
def extract_image_feature(img_path):
    base_model = InceptionV3(weights="imagenet")
    model_new = Model(base_model.input, base_model.layers[-2].output)

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = model_new.predict(x, verbose=0)
    return feature

# -------------------------------------------------
# Generate caption (Greedy Search)
# -------------------------------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    final = in_text.split()
    final = final[1:-1] if final[-1] == "endseq" else final[1:]
    return " ".join(final)

# -------------------------------------------------
# Display image and caption
# -------------------------------------------------
def show_image_with_caption(img_path, caption):
    img = image.load_img(img_path, target_size=(299, 299))
    plt.imshow(img)
    plt.axis("off")
    #plt.title(f"💬 {caption}", fontsize=14, color="darkblue", pad=10)
    plt.title(caption, fontsize=14, color="darkblue", pad=10)
    plt.show()

# -------------------------------------------------
# Main entry
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at {image_path}")

    model, tokenizer, max_length = load_tokenizer_and_model()

    print("🔍 Extracting features from image...")
    photo = extract_image_feature(image_path)

    print("🧠 Generating caption...")
    caption = generate_caption(model, tokenizer, photo, max_length)

    print(f"\n🖼️ Image: {os.path.basename(image_path)}")
    print(f"💬 Predicted caption: {caption}\n")

    show_image_with_caption(image_path, caption)

if __name__ == "__main__":
    main()
