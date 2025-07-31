# spam_streamlit_app.py

import streamlit as st
import joblib

# Load the trained model and vectorizer
try:
    model = joblib.load('Spam_mail.joblib')          # Your classifier (e.g., Naive Bayes, Logistic Regression)
    vectorizer = joblib.load('vectorizer.joblib')    # TF-IDF or CountVectorizer
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# Set page config
st.set_page_config(page_title="Spam Mail Detector", layout="centered")

# App title
st.title("📧 Spam Mail Detection using Machine Learning")

# Instructions
st.markdown("Enter the content of an email, and the model will tell you whether it's spam or not.")

# Input text box
email_text = st.text_area("✍️ Email Content:", height=200)

# Predict button
if st.button("🔍 Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter email content to analyze.")
    else:
        # Vectorize and predict
        try:
            transformed_input = vectorizer.transform([email_text])
            prediction = model.predict(transformed_input)[0]

            if prediction == 1:
                st.error("🚫 This email is likely **SPAM**.")
            else:
                st.success("✅ This email is **NOT SPAM**.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
