# scripts/train.py — Safe version for Windows (No freezing)

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import importlib.util

# -------------------------------------------------
# ✅ Enable eager execution (prevents silent hangs)
# -------------------------------------------------
tf.config.run_functions_eagerly(True)

# -------------------------------------------------
# ✅ Import load_captions directly
# -------------------------------------------------
load_path = os.path.join(os.path.dirname(__file__), "load_captions.py")
spec = importlib.util.spec_from_file_location("load_captions", load_path)
load_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_module)
load_captions = load_module.load_captions

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, "features", "image_features.pkl")
TOKENIZER_FILE = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
SEQS_FILE = os.path.join(BASE_DIR, "models", "sequences.npz")
MODEL_FILE = os.path.join(BASE_DIR, "models", "caption_model.h5")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------------------------
# Load required data
# -------------------------------------------------
print("📦 Loading features, tokenizer, and metadata...")

with open(FEATURES_FILE, "rb") as f:
    features = pickle.load(f)

with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)

meta = np.load(SEQS_FILE, allow_pickle=True)
vocab_size = int(meta["vocab_size"])
max_length = int(meta["max_len"])

captions = load_captions()
print(f"✅ Features: {len(features)} | Vocab size: {vocab_size} | Max length: {max_length}")

# -------------------------------------------------
# Train/Validation split
# -------------------------------------------------
keys = list(captions.keys())
train_keys, val_keys = train_test_split(keys, test_size=0.2, random_state=42)

train_descriptions = {k: captions[k] for k in train_keys}
val_descriptions = {k: captions[k] for k in val_keys}
train_features = {k: features[k] for k in train_keys if k in features}
val_features = {k: features[k] for k in val_keys if k in features}

print(f"📊 Train images: {len(train_descriptions)}, Val images: {len(val_descriptions)}")

# -------------------------------------------------
# Data generator (memory-safe, single-threaded)
# -------------------------------------------------
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size=64):
    """Yield one batch of (X1, X2, y) safely on Windows."""
    while True:
        X1, X2, y = [], [], []
        for key, desc_list in descriptions.items():
            photo = photos.get(key)
            if photo is None:
                continue
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield [[np.array(X1), np.array(X2)], np.array(y)]
                        X1, X2, y = [], [], []

# -------------------------------------------------
# Build CNN-LSTM model
# -------------------------------------------------
print("🧠 Building model...")

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation="relu")(decoder1)
outputs = Dense(vocab_size, activation="softmax")(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-4))
model.summary()

# -------------------------------------------------
# Callbacks
# -------------------------------------------------
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_{epoch:02d}_valLoss_{val_loss:.4f}.h5")

checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", verbose=1, save_best_only=False, mode="min"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)

# -------------------------------------------------
# Training setup (single-threaded for safety)
# -------------------------------------------------
batch_size = 64
train_steps = sum(len(d) for d in train_descriptions.values()) // batch_size
val_steps = sum(len(d) for d in val_descriptions.values()) // batch_size

train_gen = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size, batch_size)
val_gen = data_generator(val_descriptions, val_features, tokenizer, max_length, vocab_size, batch_size)

print("\n🚀 Starting training (safe mode, no multiprocessing)...\n")

history = model.fit(
    train_gen,
    epochs=20,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1,
    workers=0,
    use_multiprocessing=False,
    max_queue_size=1
)

# -------------------------------------------------
# Save final model
# -------------------------------------------------
model.save(MODEL_FILE)
print(f"\n✅ Training complete! Final model saved → {MODEL_FILE}")
print(f"📦 All checkpoints saved to → {CHECKPOINT_DIR}")
