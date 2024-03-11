# Importing necessary libraries and modules
from flask import Flask, render_template, jsonify, request
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

# Assuming 'model' is an instance of one of the sklearn models
from app import model

# Load the spaCy English language model for Natural Language Processing (NLP) tasks
nlp = spacy.load("en_core_web_sm")

# Initialize the Flask application
app = Flask(__name__)


def preprocess_text(text):
    """
    Function to preprocess text by replacing URLs and email addresses with placeholders,
    tokenizing, lemmatizing, and removing stopwords and non-alpha characters.
    """
    text = re.sub(r'https?://\S+|www\.\S+', ' urlplaceholder ', text)
    text = re.sub(r'\S*@\S*\s?', ' emailplaceholder ', text)
    doc = nlp(text)
    clean_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(clean_tokens)


def load_and_preprocess_data(filename='data.csv'):
    """
    Function to load and preprocess data from a CSV file. It applies text preprocessing
    to each 'text' column entry and ensures the 'label' column is of integer type.
    """
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)
    return dataset


def train_model(dataset, model_type='logistic_regression'):
    """
    Function to train a machine learning model based on the preprocessed dataset.
    It supports logistic regression, random forest, and gradient boosting models.
    Model selection is based on the 'model_type' parameter.
    """
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100)

    pipeline = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 3)), model)
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000],
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
    }

    # Additional hyperparameters for the models can be specified here
    if model_type == 'logistic_regression':
        parameters['logisticregression__C'] = [0.1, 1, 10]

    cv = GridSearchCV(pipeline, parameters, cv=5)
    cv.fit(X_train, y_train)
    print(f"Best parameters: {cv.best_params_}")
    return cv.best_estimator_


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Flask route to handle the homepage requests. It supports both GET (render the form)
    and POST (process the form data and return the phishing detection result).
    """
    if request.method == 'POST':
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            subject = data['subject']
            message = data['message']
            combined_input = f"Subject: {subject}\nMessage: {message}"
            preprocessed_text = preprocess_text(combined_input)
            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100
            return jsonify(probability="{:.2f}".format(phishing_prob))
        else:
            user_input = request.form.get('text')
            preprocessed_text = preprocess_text(user_input)
            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100
            return render_template('index.html', probability="{:.2f}%".format(phishing_prob))
    return render_template('index.html')


if __name__ == '__main__':
    # Initialize and train the model here if needed, or load a pre-trained model
    app.run(debug=True)
