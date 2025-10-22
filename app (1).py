import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


st.title("Fake News Classification App")

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Create input fields and a button
news_article = st.text_area("Enter the news article here:")
classify_button = st.button("Classify")

# Define the stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Implement prediction logic
if classify_button:
    if news_article:
        # Preprocess the input text
        preprocessed_text = stemming(news_article)

        # Transform the preprocessed text using the vectorizer
        transformed_text = vectorizer.transform([preprocessed_text])

        # Make a prediction
        prediction = model.predict(transformed_text)

        # Display the result
        if prediction[0] == 0:
            st.success("Classification: Real News")
        else:
            st.error("Classification: Fake News")
    else:
        st.warning("Please enter a news article to classify.")
