import streamlit as st
import pickle
import string
import joblib
import re
import base64

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('mlp_classifier_model.pkl')
tfidf_v = joblib.load('tfidf_vectorizer.pkl')

# Function to set background image from a local file
def set_bg_from_local(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-attachment: fixed;
            background-size: cover;
        }}
        .main-title {{
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }}
        .result-text {{
            font-size: 26px;
            font-weight: bold;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_bg_from_local("D:/NLP/Fake_News_img.jpg")

# Text preprocessing function
def preprocess_text(text):
    text = text.replace('â€™', "'").replace('â€"', '-').replace('â€œ', '"').replace('â€', '"')
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Streamlit UI
st.markdown('<h1 class="main-title" style="color:white;">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:white; font-size:18px; text-align: center;">Enter a news article or statement to check if it\'s <em>Real</em> or <em>Fake</em>.</p>', unsafe_allow_html=True)


user_input = st.text_area("Input Text", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_text = preprocess_text(user_input)
        vectorized = tfidf_v.transform([processed_text])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.markdown('<div class="result-text" style="color:green;">✅ This text appears to be <strong>Real</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-text" style="color:red;">⚠️ This text appears to be <strong>Fake</strong>.</div>', unsafe_allow_html=True)