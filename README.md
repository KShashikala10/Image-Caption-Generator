
# 🧠 Image Caption Generator

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-FF1493)
![Model](https://img.shields.io/badge/Model-InceptionV3%20%2B%20LSTM-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Developed%20On-Windows%2010-lightgrey?logo=windows)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blueviolet)

An end-to-end **Image Caption Generator** that automatically generates meaningful captions for images using **Deep Learning**, combining **InceptionV3 (CNN)** for feature extraction and **LSTM** for text generation.  
This project is trained on the **Flickr8k dataset** and includes a full training + inference pipeline.

---

## 🚀 Features

- ✔️ Pretrained **InceptionV3** for visual feature extraction  
- ✔️ Custom **LSTM-based decoder** for caption generation  
- ✔️ Clean code architecture with modular scripts  
- ✔️ Supports **custom images** for captioning  
- ✔️ Displays image + caption using matplotlib  
- ✔️ Includes BLEU score evaluation  
- ✔️ Professional project structure for GitHub/Resume  

---

## 📂 Project Structure

```

Image_Caption_Generator/
│
├── data/                           # NOT uploaded (dataset)
│   ├── Flickr8k_Dataset/
│   └── Flickr8k_text/
│
├── features/                       # NOT uploaded (large)
│   └── image_features.pkl
│
├── models/
│   ├── tokenizer.pkl               # uploaded
│   ├── sequences.npz               # uploaded
│   └── caption_model.h5            # optional upload (large)
│
├── scripts/
│   ├── extract_features.py
│   ├── load_captions.py
│   ├── create_tokenizer_and_sequences.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate_bleu.py
│
├── examples/                       # small demo output
│   ├── sample1.jpg
│   └── result_sample1.png
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore

````

---

## ⚙️ Installation & Setup

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/<your-username>/Image_Caption_Generator.git
cd Image_Caption_Generator
````

### **2️⃣ Create and activate virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

### **3️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

### **4️⃣ Download Flickr8k dataset**

Download from Kaggle:
🔗 [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Place as:

```
data/
 ├── Flickr8k_Dataset/
 └── Flickr8k_text/
```

---

## 🧠 Training the Model

### ✔️ Step 1 — Extract features from images

```bash
python scripts/extract_features.py
```

### ✔️ Step 2 — Create tokenizer + sequences

```bash
python scripts/create_tokenizer_and_sequences.py
```

### ✔️ Step 3 — Train the captioning model

```bash
python scripts/train.py
```

After training:

```
📦 Final model saved → models/caption_model.h5
```

---

## 🧪 Generate Captions (Inference)

### Caption for dataset image:

```bash
python scripts/inference.py --image data/Flickr8k_Dataset/1000268201_693b08cb0e.jpg
```

### Caption for your own image:

```bash
python scripts/inference.py --image "C:\Users\Admin\Desktop\my_image.jpg"
```

The program will display:

* the input image
* the generated caption

---

## 📦 Requirements

```
tensorflow==2.15.0
numpy
matplotlib
pandas
scikit-learn
tqdm
nltk
```

---

## 🌟 Future Improvements

* Add **Beam Search** decoding
* Add **Attention Mechanism** (Show, Attend & Tell architecture)
* Use **MS COCO dataset** for improved accuracy
* Build **Flask/Streamlit web app**
* Deploy model on cloud

---

## 👩‍💻 Author

**K. Shashikala**
🔗 LinkedIn: [https://www.linkedin.com/in/k-shashikala10)
🐙 GitHub: [https://github.com/KShashikala10)

---

## 📝 License

This project is licensed under the **MIT License**.


