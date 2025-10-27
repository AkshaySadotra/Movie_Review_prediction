import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model



# load the iddb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


model = load_model('simplernn_imdb_model.h5')

# helper function to decode review
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

# function to preprocess the uer input review
def preprocess_text(review):
    words = review.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded


def predict_review(review):
    processed_input = preprocess_text(review)
    prediction = model.predict(processed_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]


import streamlit as st
# streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative).") 
user_review = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    preprocessed_input_value = preprocess_text(user_review)
    prediction = model.predict(preprocessed_input_value)
    
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    st.write(f'Sentiment: {sentiment}')
    st.write(f' Prediction Score: {prediction[0][0]}')
else:
    st.write("Please enter a review and click the 'Predict Sentiment' button.")    
