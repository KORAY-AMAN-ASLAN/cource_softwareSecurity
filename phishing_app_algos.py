"""
This Python script is designed to create a web application using Flask that classifies email text as either legitimate,
 or phishing.
It utilizes machine learning models that have been pre-trained to identify characteristics of phishing attempts within,
email content.

 Authour: Koray Aman Arabzadeh
 Cource: Mjukvarusäkerhet, Software Security ,  Mittuniversitetet
 Year: 2024-04-01

Video can be found here: https://www.youtube.com/watch?v=-3TfoUi6oTk
"""


# Import necessary libraries

import os
import warnings
import joblib  # For model persistence, allowing saving and loading of the ML model.
from flask import Flask, render_template, request, \
    jsonify  # Flask for web app functionality, template rendering, and request handling.
import pandas as pd  # Pandas for efficient data manipulation and analysis.
import spacy  # spaCy for natural language processing tasks.
from sklearn.metrics import classification_report  # Scikit-learn for performance metrics.
from sklearn.model_selection import train_test_split, \
    GridSearchCV  # For splitting data and optimizing model parameters.
from sklearn.pipeline import make_pipeline  # For creating a sequence of data transformations ending in an estimator.
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical data through TF-IDF.
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # ML models.
from sklearn.linear_model import LogisticRegression  # Another ML model option.
import re  # For regex operations, useful in text preprocessing.
from sklearn.svm import SVC  # Support Vector Classifier for another ML model option.

# Attempt to load the model's path from environment variable; use a default if not found.
MODEL_PATH = os.getenv('MODEL_PATH', 'logistic_regression_model.pkl')

# Load the English language model for spaCy, used for NLP tasks.
nlp = spacy.load("en_core_web_sm")

# Initialize the Flask app.
app = Flask(__name__)


# Function to preprocess input text for NLP analysis.
def preprocess_text(text, replace_urls=True, replace_emails=True, remove_non_alpha=True, lower_case=True,
                    remove_stopwords=True):
    """
    Cleans and normalizes text data, preparing it for NLP processing and ML modeling. It replaces URLs and emails,
    optionally removes non-alphanumeric characters and stopwords, and normalizes the casing.
    """
    # Replace URLs and emails with placeholders to neutralize potential noise in text data.
    if replace_urls:
        text = re.sub(r'https?://\S+|www\.\S+', 'urlplaceholder', text)
    if replace_emails:
        text = re.sub(r'\S*@\S*\s?', 'emailplaceholder', text)

    doc = nlp(text)
    clean_tokens = [token.lemma_.lower() if lower_case else token.lemma_ for token in doc if
                    (not remove_stopwords or not token.is_stop) and (not remove_non_alpha or token.is_alpha)]
    print(clean_tokens)
    return " ".join(clean_tokens)


# Load data from a CSV file, apply preprocessing, and prepare it for model training.
def load_and_preprocess_data(filename='data.csv'):
    """
    Loads phishing data from a CSV, applies text preprocessing, and ensures the label column is integer-typed.
    """
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)
    print(dataset)
    class_disribution = dataset['label'].value_counts(normalize=True) * 100;
    print("Class distribution in dataset (%): ") # min dataset är balancerat
    print(class_disribution)
    return dataset

# Train a machine learning model using the preprocessed dataset.
def train_model(dataset, model_type='logistic_regression'):

    # Split the dataset into training and testing sets to evaluate the model on unseen data.
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defines a dictionary mapping model type names to their corresponding initialized model objects.
    models = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'svm': SVC(probability=True)  # Enables probability estimation for SVM, necessary for some metrics and operations.
    }

    # Checks if the specified model type is available, then constructs a pipeline coupling TfidfVectorizer with the chosen model.
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}.")
    model = models[model_type]
    pipeline = make_pipeline(TfidfVectorizer(), model)

    # Configuration of general and model-specific hyperparameters for optimization using GridSearchCV.
    parameters = {


        'tfidfvectorizer__max_features': [3000, 5000, 7000],
 
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)]
    }
    model_params = {
        'logistic_regression': {'logisticregression__C': [0.1, 1, 10]},


        'random_forest': {'randomforestclassifier__n_estimators': [100, 200]},

   
        'gradient_boosting': {'gradientboostingclassifier__n_estimators': [100, 200]},

        'svm': {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
    }

    parameters.update(model_params[model_type])


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        cv = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')

        cv.fit(X_train, y_train)


    print(f"Best parameters: {cv.best_params_}")
    y_pred = cv.predict(X_test)
    print(classification_report(y_test, y_pred))

    return cv.best_estimator_


# Define the behavior for the root path ("/"), handling both GET and POST requests.
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles web requests to the root path. For POST requests, it processes submitted text data and
    returns the model's phishing probability prediction. For GET requests, it simply renders the homepage.
    """
    if request.method == 'POST':
        try:
            # Determine if the request is JSON or form data and extract text data accordingly.
            data = request.get_json() if request.is_json else request.form
            combined_input = f"Subject: {data.get('subject', '')}\nMessage: {data.get('message', '')}"
            preprocessed_text = preprocess_text(combined_input)

            # Attempt to load the trained model and predict the phishing probability of the input text.
            try:
                model = joblib.load(MODEL_PATH)
            except FileNotFoundError:
                return jsonify(error="Model not found."), 500

            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100

            return jsonify(probability="{:.2f}".format(phishing_prob)) if request.is_json else render_template(
                'index.html', probability="{:.2f}%".format(phishing_prob))
        except Exception as e:
            # Handle unexpected errors gracefully.
            return jsonify(error=str(e)), 500

    # Render the homepage for GET requests.
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
