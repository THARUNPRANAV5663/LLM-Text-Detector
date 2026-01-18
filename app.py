import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load TF-IDF vectorizer
with open("tfidf_tokenizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load trained model
model = load_model("text_classification_model.h5")

# Streamlit UI
st.title("LLM Text Detector 🚦")
st.write("Classify text as **Human-written** or **AI-generated** with confidence scores.")

# Threshold slider with clear explanation
st.markdown("### Decision Threshold")
threshold = st.slider(
    "Move the bar (0.0 = Human, 1.0 = AI)",
    0.0, 1.0, 0.7, 0.01
)
st.caption(
    "👉 At **0.0**, everything is classified as Human.\n"
    "👉 At **1.0**, everything is classified as AI.\n"
    "👉 Middle values (like 0.5–0.7) balance sensitivity."
)

# Text input
user_input = st.text_area("✍️ Enter text to classify:", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        X_input = tfidf_vectorizer.transform([user_input]).toarray()

        # Predict probability
        prediction_prob = model.predict(X_input)[0][0]
        prediction = int(prediction_prob > threshold)

        # Show classification
        if prediction == 0:
            st.success(f"🟢 Classification: Human-written\nConfidence: {1 - prediction_prob:.4f}")
            st.progress(int((1 - prediction_prob) * 100))
        else:
            st.error(f"🔴 Classification: AI-generated\nConfidence: {prediction_prob:.4f}")
            st.progress(int(prediction_prob * 100))