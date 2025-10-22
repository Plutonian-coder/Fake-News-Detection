AI-Powered News Verifier

This Streamlit application provides a robust, dual-analysis approach to news verification. It helps users assess the credibility of a news article by combining a pre-trained local machine learning model with the advanced, real-time fact-checking capabilities of the Google Gemini API.

When a user submits an article, the app displays a side-by-side comparison:

Local Model: A classification (Real/Fake) from a LogisticRegression model.

Gemini Analysis: A detailed, reasoned explanation from Gemini, complete with a final classification and links to the web sources it used for fact-checking.

✨ Features

Dual-Model Analysis: Compares the prediction of a static, pre-trained model with a dynamic, LLM-based analysis from Gemini.

Gemini Advanced Fact-Checking: Leverages gemini-2.5-flash with Google Search grounding to cross-reference the article against live web data.

Source Citation: Displays the web sources Gemini used to reach its conclusion, allowing for further user verification.

Side-by-Side Comparison: Presents both classifications in a clear, two-column layout.

Robust API Handling: Implements an exponential backoff retry strategy for resilient calls to the Gemini API.

🛠️ How It Works

User Input: The user pastes the full text of a news article into the text area.

Analysis Triggered: When the "Analyze Article" button is clicked, two processes run in parallel:

Local Model (Column 1): The text is preprocessed using nltk (stemming, stopword removal, regex cleaning). This cleaned text is then transformed by a pre-trained TfidfVectorizer and fed into a LogisticRegression model, which classifies it as "Real" or "Fake" based on its training data.

Gemini Model (Column 2): The raw article text is sent to the Gemini API. A system prompt instructs the model to act as an expert fact-checker, use Google Search, and provide a concise, one-paragraph explanation for its reasoning, followed by a final classification.

Display Results: The app displays a spinner while fetching the analyses, then presents both results in their respective columns. The Gemini analysis includes any web sources cited.

🚀 Tech Stack

Python 3.x

Streamlit: For the web application UI.

Scikit-learn: For the local machine learning model (LogisticRegression, TfidfVectorizer).

Joblib: For loading the pre-trained .joblib model files.

NLTK: For natural language text preprocessing (stemming, stopwords).

Requests: For making HTTP calls to the Gemini API.

⚙️ Setup & Installation

To run this application locally, follow these steps:

Clone the repository:

git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name


Create and activate a virtual environment:

python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate


Install the required packages:

pip install -r requirements.txt


Obtain Model Files:
This project requires two pre-trained model files, which are not included in this repository:

logistic_regression_model.joblib

tfidf_vectorizer.joblib
You must have these files (from your model training script) in the same directory as app.py.

NLTK Data:
The application will automatically attempt to download the stopwords corpus from NLTK on its first run.

🚨 IMPORTANT: API Key Security

The app.py file currently has a hardcoded API key:

apiKey = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE" 


This is a major security risk. Never commit visible API keys to a public or private repository.

How to Fix: Use Streamlit's Secrets Management.

Create a secrets file:
In your project's root, create a directory .streamlit and a file secrets.toml inside it.

.
├── .streamlit/
│   └── secrets.toml
├── app.py
└── ...


Add your key to secrets.toml:

# .streamlit/secrets.toml
GEMINI_API_KEY = "your-real-api-key-goes-here"


Update app.py to use the secret:
Find this line:

apiKey = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE" 


And replace it with this:

apiKey = st.secrets.get("GEMINI_API_KEY")


Add .streamlit to your .gitignore file to ensure your secrets file is never committed.

▶️ Running the App

Once your environment is set up, your model files are in place, and your API key is secured, you can run the app:

streamlit run app.py


Your app will be available at http://localhost:8501.