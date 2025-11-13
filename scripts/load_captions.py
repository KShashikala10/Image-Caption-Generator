# scripts/load_captions.py
import os

def load_captions():
    """
    Loads and cleans Flickr8k captions from Flickr8k.token.txt.
    Returns a dictionary mapping image IDs -> list of captions.
    Each caption is surrounded by 'startseq' and 'endseq' tokens.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CAPTION_FILE = os.path.join(BASE_DIR, "data", "Flickr8k_text", "Flickr8k.token.txt")

    if not os.path.exists(CAPTION_FILE):
        raise FileNotFoundError(
            f"❌ Caption file not found at: {CAPTION_FILE}\n"
            f"👉 Please ensure Flickr8k_text folder exists inside the 'data' directory."
        )

    captions = {}
    print(f"📘 Loading captions from: {CAPTION_FILE}")

    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if len(tokens) < 2:
                continue

            image_id, caption = tokens
            image_id = image_id.split(".")[0]  # remove .jpg
            caption = caption.lower().strip()
            caption = f"startseq {caption} endseq"

            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)

    print(f"✅ Loaded captions for {len(captions)} images.")
    return captions
