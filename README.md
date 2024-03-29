# SentimentAnalyzerApp

Welcome to the SentimentAnalyzerApp repository! This project is a simple web application for sentiment analysis of restaurant reviews. It utilizes a Bag of Words (BoW) model and a pre-trained classifier to predict whether a given review expresses positive or negative sentiment.

## Getting Started

To get started with the SentimentAnalyzerApp, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Srish0218/SentimentAnalyzerApp.git
   ```

2. Navigate to the project directory:

   ```bash
   cd SentimentAnalyzerApp
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   Open your web browser and go to [http://localhost:8501](http://localhost:8501) to use the Sentiment Analyzer App.

## Usage

1. Enter a restaurant review in the provided text area.
2. Click the "Predict Sentiment" button to see the predicted sentiment (positive or negative).

## Models

- The Bag of Words (BoW) model is loaded from the `c1_BoW_Sentiment_Model.pkl` file.
- The classifier model is loaded from the `c2_Classifier_Sentiment_Model` file.

## Preprocessing

- Input reviews are preprocessed by converting to lowercase, removing non-alphabetic characters, stemming, and removing English stopwords.

## Contributing

If you'd like to contribute to this project, please follow the standard GitHub fork and pull request workflow. Feel free to open issues for bug reports or feature requests.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or suggestions, feel free to reach out to the project owner:

- GitHub: [https://github.com/Srish0218](https://github.com/Srish0218)

Thank you for using the SentimentAnalyzerApp!