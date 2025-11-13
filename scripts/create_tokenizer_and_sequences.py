# scripts/create_tokenizer_and_sequences.py
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CAPTION_FILE = os.path.join(BASE_DIR, "data", "Flickr8k_text", "Flickr8k.token.txt")
TOKENIZER_FILE = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
SEQS_FILE = os.path.join(BASE_DIR, "models", "sequences.npz")

# -------------------------------------------------
# Verify captions file exists
# -------------------------------------------------
if not os.path.exists(CAPTION_FILE):
    raise FileNotFoundError(f"❌ Caption file not found: {CAPTION_FILE}\n"
                            "👉 Make sure Flickr8k_text folder is inside the 'data' directory.")

# -------------------------------------------------
# Load captions
# -------------------------------------------------
def load_captions(filename):
    print(f"📘 Loading captions from: {filename}")
    captions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens
            image_id = image_id.split('.')[0]
            caption = f"startseq {caption.lower()} endseq"
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    print(f"✅ Total image IDs loaded: {len(captions)}")
    return captions

captions = load_captions(CAPTION_FILE)

# -------------------------------------------------
# Flatten all captions into a list
# -------------------------------------------------
all_captions = []
for key in captions.keys():
    all_captions.extend(captions[key])
print(f"✅ Total captions collected: {len(all_captions)}")

# -------------------------------------------------
# Create and fit tokenizer
# -------------------------------------------------
print("🧠 Creating tokenizer...")
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
print(f"✅ Vocabulary size: {vocab_size}")

# -------------------------------------------------
# Find maximum caption length
# -------------------------------------------------
max_length = max(len(caption.split()) for caption in all_captions)
print(f"✅ Max caption length: {max_length}")

# -------------------------------------------------
# Save tokenizer and metadata
# -------------------------------------------------
os.makedirs(os.path.dirname(TOKENIZER_FILE), exist_ok=True)
with open(TOKENIZER_FILE, "wb") as f:
    pickle.dump(tokenizer, f)

np.savez(SEQS_FILE, vocab_size=vocab_size, max_len=max_length)
print(f"📦 Saved tokenizer → {TOKENIZER_FILE}")
print(f"📦 Saved metadata  → {SEQS_FILE}")

print("🎉 Caption data preparation complete!")
