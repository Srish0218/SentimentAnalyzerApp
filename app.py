import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

# Load BoW model
cv_file = 'c1_BoW_Sentiment_Model.pkl'  # Replace with the actual path
cv = pickle.load(open(cv_file, "rb"))

# Load classifier model
classifier_file = 'c2_Classifier_Sentiment_Model'  # Replace with the actual path
classifier = joblib.load(classifier_file)

# Load English stopwords
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')  # Remove 'not' from stopwords

ps = PorterStemmer()

st.title("Sentiment Analysis Web App")

# Add a textarea for user input
user_input = st.text_area("Enter a restaurant review:")

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess input
        review = re.sub('[^a-zA-Z]', ' ', user_input)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)

        # Transform using BoW model
        input_transformed = cv.transform([review]).toarray()

        # Make prediction
        prediction = classifier.predict(input_transformed)

        # Display result
        Sentiment = prediction[0]
        if prediction[0] == 1:
            st.write("Predicted Sentiment: Positive sentiment")
        else:
            st.write("Predicted Sentiment: Negative sentiment")
    else:
        st.warning("Please enter a review.")
