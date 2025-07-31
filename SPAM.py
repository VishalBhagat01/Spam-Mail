# -*- coding: utf-8 -*-
"""
Spam Mail Prediction Web App using Streamlit
"""

import joblib
import streamlit as st

# Load the trained model and vectorizer
model = joblib.load('Spam_mail.joblib')          # Classifier
vectorizer = joblib.load('vectorizer.joblib')# TfidfVectorizer

# App title
st.set_page_config(page_title="Spam Mail Detection", layout="centered")
st.title("📧 Spam Mail Prediction using Machine Learning")

# Input box for email content
email_text = st.text_area("Enter the email content below:", height=200)

# Initialize prediction result
result = ""

# Predict button
if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter some email content.")
    else:
        # Vectorize the input text
        transformed_text = vectorizer.transform([email_text])
        
        # Predict using the loaded model
        prediction = model.predict(transformed_text)[0]
        
        if prediction == 1:
            result = "⚠️ This is likely **SPAM**."
            st.error(result)
        else:
            result = "✅ This is **NOT spam**."
            st.success(result)
