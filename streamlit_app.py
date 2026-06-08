
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded (only need to do this once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt') # Changed from punkt_tab to punkt for broader compatibility
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Load the model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, tfidf_vectorizer = load_artifacts()

# Initialize lemmatizer and stopwords once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function (replicated from notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    return text

# Text preprocessing function
def preprocess_text(text):
    cleaned_text = clean_text(text)
    tokens = nltk.word_tokenize(cleaned_text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Streamlit App Layout
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below to predict its sentiment (positive/negative).')

# Text input from user
user_input = st.text_area('Enter your movie review here:', height=200)

if st.button('Predict Sentiment'):
    if user_input:
        # Preprocess the input
        processed_review = preprocess_text(user_input)

        # Transform the review using the loaded TF-IDF vectorizer
        # Need to put the processed_review in a list for vectorizer.transform
        review_vectorized = tfidf_vectorizer.transform([processed_review])

        # Make prediction
        prediction = model.predict(review_vectorized)

        # Display result
        if prediction[0] == 1:
            st.success('Sentiment: Positive 👍')
        else:
            st.error('Sentiment: Negative 👎')
    else:
        st.warning('Please enter a review to predict sentiment.')
