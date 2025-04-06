import streamlit as st
import joblib

# Load the vectorizer and model
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Streamlit app title
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real.")

# User input
news_input = st.text_area("News Article:", "")

# Button to check news
if st.button("Check News"):
    if news_input.strip():  # Check if input is not empty
        transform_input = vectorizer.transform([news_input])  # Transform input text
        prediction = model.predict(transform_input)  # Get prediction

        # Display result
        if prediction[0] == 1:
            st.success("The News is Real! ✅")
        else:
            st.error("The News is Fake! ❌")
    else:
        st.warning("Please enter some text to analyze.")
