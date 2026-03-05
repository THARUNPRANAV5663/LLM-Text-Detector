# 🤖 LLM Text Detector

A deep learning web application that classifies whether a given text is **Human-written** or **AI-generated**, with real-time confidence scores and an adjustable decision threshold.

🔗 **Live Demo:** https://llm-text-detector-kstp.streamlit.app/

---

## 📌 Problem Statement

With the rapid rise of AI-generated content, distinguishing between human and AI-written text has become critical for academic integrity, journalism, and content moderation. This project tackles that challenge using a TF-IDF + Neural Network pipeline.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **ML/DL:** TensorFlow 2.20, Keras — Sequential Neural Network
- **NLP:** scikit-learn TF-IDF Vectorizer (3000 features)
- **Deployment:** Streamlit

---

## 📊 Dataset

- Combined two datasets:
  - `train_v2_drcat_02.csv` — essay dataset with AI/human labels
  - `Training_Essay_Data.csv` — additional essay training data
- **Target:** Human-written (0) / AI-generated (1)
- Class imbalance handled using **computed class weights**

---

## 🧠 Model Architecture

```
Input (3000 TF-IDF features)
→ Dense(128, ReLU) → Dropout(0.3)
→ Dense(64, ReLU)  → Dropout(0.3)
→ Dense(1, Sigmoid)
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping (patience=3), ModelCheckpoint
- **Decision Threshold:** 0.7 (stricter, reduces false positives)

---

## ⚙️ Workflow

1. Load and combine two essay datasets
2. Drop nulls and unnecessary columns
3. TF-IDF vectorization (max 3000 features)
4. Stratified train/test split (80/20, random_state=144)
5. Compute class weights to handle imbalance
6. Train neural network with EarlyStopping and ModelCheckpoint
7. Save best model (`best_model.h5`) and TF-IDF vectorizer (`tfidf_tokenizer.pkl`)
8. Streamlit app with adjustable threshold slider for real-time classification

---

## ✨ App Features

- Paste any text and classify it instantly
- **Confidence score** displayed for each prediction
- **Adjustable threshold slider** (0.0 to 1.0) to tune sensitivity
- Visual progress bar showing confidence level

---

## 📁 Project Structure

```
LLM-Text-Detector/
├── app.py                        # Streamlit app
├── train_model.py                # Model training script
├── best_model.h5                 # Best saved model (via ModelCheckpoint)
├── text_classification_model.h5  # Final trained model
├── tfidf_tokenizer.pkl           # Fitted TF-IDF vectorizer
├── requirements.txt
└── runtime.txt                   # Python 3.10
```

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/THARUNPRANAV5663/LLM-Text-Detector.git
cd LLM-Text-Detector
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Tharun Pranav K S**  
[LinkedIn](https://www.linkedin.com/in/tharunpranav-k-s-8a0608250/) | [GitHub](https://github.com/THARUNPRANAV5663)
