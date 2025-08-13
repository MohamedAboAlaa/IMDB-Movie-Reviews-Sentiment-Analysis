import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Load trained model and vectorizer
# -------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter your review, and the app will predict whether it is positive or negative:")

# Input text
user_input = st.text_area("Review:")

if st.button("Analyze Review"):
    if user_input.strip() != "":
        cleaned_input = clean_text(user_input)
        vect_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vect_input)[0]
        sentiment = "Positive 😃" if prediction == 1 else "Negative 😞"
        st.success(f"Classification: {sentiment}")
    else:
        st.error("Please enter a review")

# -------------------------------
# GitHub link with icon
# -------------------------------
github_url = "https://github.com/YOUR_USERNAME/YOUR_REPO"
st.markdown(
    f'[![GitHub](https://img.shields.io/badge/GitHub-Visit%20Repo-black?logo=github)]({github_url})',
    unsafe_allow_html=True
)
