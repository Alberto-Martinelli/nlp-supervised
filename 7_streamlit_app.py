import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def rating_prediction(input_review):
    df = pd.read_csv('data_cleaned.csv')
    df = df.dropna(subset=['score'])

    df['review_cleaned_joined'] = df['review_cleaned_joined'].fillna("").astype(str)
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(df['review_cleaned_joined'])
    
    loaded_log_reg = joblib.load('logistic_regression_model.pkl')

    review = [input_review]

    # Transform the example review using the TF-IDF vectorizer
    review_tfidf = tfidf.transform(review)

    # Predict the rating using the loaded logistic regression model
    predicted_rating = loaded_log_reg.predict(review_tfidf)

    # print(f"Predicted Rating for the example review using the loaded model: {predicted_rating[0]}")
    return predicted_rating

# Function to predict sentiment for a randomly generated review
def predict_sentiment(review, tokenizer, model):
    # Text Preprocessing (tokenization and padding)
    review_sequence = tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_sequence, padding='post', maxlen=50)

    # Predict sentiment (0 for negative, 1 for positive)
    prediction = model.predict(review_padded)
    sentiment = "positive" if prediction >= 0.5 else "negative"
    return sentiment

def sentiment_prediction(review):
    df = pd.read_csv('sentiment_analysis_results.csv')
    model_save_path = "sentiment_analysis_model.h5"
    model = load_model(model_save_path)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['review_cleaned_joined'])  # Ensure the tokenizer is fit on the same training data
    sentiment = predict_sentiment(review, tokenizer, model)
    return sentiment



# Title
st.title("NLP Project 2 - Supervised")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ("Rating prediction", "Sentiment analysis"))

# Home section
if section == "Rating prediction":
    st.header("Rating prediction")
    # st.write("This is the home page of your Streamlit app. You can navigate to other sections using the sidebar.")
    
    review = st.text_area("Enter a review for prediction:")

    if st.button("Predict Rating"):
        prediction = rating_prediction(review)  # Replace with your model inference logic
        st.write(f"Predicted Rating: {prediction[0]}")

# Section 1
elif section == "Sentiment analysis":
    st.header("Sentiment analysis")
    # st.write("This is the content for Section 1. You can add any content you like here, such as text, charts, or images.")

    # User Input
    review = st.text_area("Enter a review for sentiment analysis:")

    # Model Prediction
    if st.button("Predict Sentiment"):
        prediction = sentiment_prediction(review)  # Replace with your model inference logic
        st.write(f"Predicted Sentiment: {prediction}")