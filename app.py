import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests
import time
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- Page Configuration ---
st.set_page_config(
    page_title="AI News Verifier",
    page_icon="📰",
    layout="wide"
)

# --- NLTK Stopwords Download ---
# We do this once at the start
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- System Prompt for Gemini ---
# This guides Gemini to act as a fact-checker
GEMINI_SYSTEM_PROMPT = """
You are an expert fact-checker. Your goal is to determine if the provided news article
is likely real or fake. Analyze the content, cross-reference it with real-world information
using your search tool, and provide a concise, one-paragraph explanation for your reasoning.
Conclude your explanation with the final classification on its own line.
Example:
Classification: Real
"""

# --- Model Loading (Cached) ---
# Use st.cache_resource to load models only once
@st.cache_resource
def load_models():
    try:
        model = joblib.load('logistic_regression_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files (logistic_regression_model.joblib or tfidf_vectorizer.joblib) not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, vectorizer = load_models()

# --- Local Model Preprocessing ---
port_stem = PorterStemmer()

def stemming(content):
    """Preprocesses text for the local Logistic Regression model."""
    if not isinstance(content, str):
        return ""
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# --- Gemini API Call Function ---
def get_gemini_analysis(article_text):
    """
    Calls the Gemini API with Google Search grounding for fact-checking.
    Implements exponential backoff for robustness.
    """
    # NOTE: The API key is an empty string. Canvas will handle authentication.
    apiKey = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE" 
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

    payload = {
        "contents": [{"parts": [{"text": article_text}]}],
        "tools": [{"google_search": {}}],  # Enable Google Search grounding
        "systemInstruction": {
            "parts": [{"text": GEMINI_SYSTEM_PROMPT}]
        },
    }

    # Setup robust session with retries (exponential backoff)
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Factor for exponential delay (1s, 2s, 4s, ...)
        status_forcelist=[429, 500, 502, 503, 504], # Statuses to retry on
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    try:
        response = session.post(
            apiUrl,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=120  # 120-second timeout for the request
        )
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status() 
        
        result = response.json()
        candidate = result.get("candidates", [{}])[0]
        content_parts = candidate.get("content", {}).get("parts", [{}])
        
        # 1. Extract the generated text
        text = content_parts[0].get("text", "Error: No text response from Gemini.")

        # 2. Extract grounding sources
        sources = []
        grounding_metadata = candidate.get("groundingMetadata", {})
        if grounding_metadata and "groundingAttributions" in grounding_metadata:
            sources = [
                {
                    "uri": attr.get("web", {}).get("uri"),
                    "title": attr.get("web", {}).get("title"),
                }
                for attr in grounding_metadata["groundingAttributions"]
                if attr.get("web")
            ]
        
        return text, sources

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, or bad HTTP responses
        error_message = f"API Request Error: {e}"
        if e.response:
            try:
                error_message = f"API Error: {e.response.status_code} - {e.response.text}"
            except Exception:
                pass # Keep the simpler error message
        return error_message, []
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        # Handle unexpected JSON structure from Gemini
        return f"Error parsing Gemini response: {e}. Response: {response.text}", []

# --- Streamlit App UI ---
st.title("📰 AI-Powered News Verifier")
st.markdown(
    "Enter a news article to get two AI-powered classifications: one from a local model and one from Gemini's advanced, real-time fact-checking."
)

st.divider()

news_article = st.text_area(
    "Enter the full news article text here:",
    height=250,
    placeholder="Paste the article content..."
)

if st.button("Analyze Article", type="primary", use_container_width=True):
    if news_article:
        with st.spinner("Analyzing... This may take a moment as Gemini searches the web."):
            # --- 1. Local Model Prediction ---
            try:
                preprocessed_text = stemming(news_article)
                transformed_text = vectorizer.transform([preprocessed_text])
                prediction = model.predict(transformed_text)
                local_result = "Real News" if prediction[0] == 0 else "Fake News"
            except Exception as e:
                local_result = f"Error: {e}"

            # --- 2. Gemini API Prediction ---
            gemini_text, gemini_sources = get_gemini_analysis(news_article)

        st.divider()
        st.subheader("Analysis Results")

        col1, col2 = st.columns(2, gap="medium")

        # --- Display Local Model Result ---
        with col1:
            st.markdown("#### 🤖 **Local Model**")
            st.markdown(
                "Based on its training data, this model classified the article as:"
            )
            if "Real" in local_result:
                st.success(f"**Classification: {local_result}**")
            elif "Fake" in local_result:
                st.error(f"**Classification: {local_result}**")
            else:
                st.warning(local_result) # Show error

        # --- Display Gemini Result ---
        with col2:
            st.markdown("#### ✨ **Gemini Advanced Analysis**")
            st.markdown(
                "Using real-time Google Search, Gemini provided this analysis:"
            )
            
            # Display Gemini's explanation
            st.info(gemini_text) 
            
            # Display sources if any were found
            if gemini_sources:
                st.markdown("##### Sources Found:")
                for source in gemini_sources:
                    if source['title'] and source['uri']:
                        st.markdown(f"- [{source['title']}]({source['uri']})")

    else:
        st.warning("Please enter a news article to analyze.")
