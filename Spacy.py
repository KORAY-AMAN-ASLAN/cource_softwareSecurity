# Flask is a micro web framework for Python, allowing you to build web applications easily.
from flask import Flask, render_template, request

# pandas is a library for data manipulation and analysis, particularly offering data structures and operations for manipulating numerical tables and time series.
import pandas as pd

# re is the regular expression library in Python, used for matching text patterns.
import re

# spaCy is an open-source library for advanced Natural Language Processing (NLP) in Python. It's designed for practical NLP tasks with pre-trained models.
import spacy

# TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features. It's part of scikit-learn, a machine learning library.
from sklearn.feature_extraction.text import TfidfVectorizer

# LogisticRegression is a linear model for classification rather than regression. It is also part of scikit-learn.
from sklearn.linear_model import LogisticRegression

# train_test_split is a function in scikit-learn to split datasets into training and testing sets.
# GridSearchCV is used for hyperparameter tuning, finding the best parameters for a model.
from sklearn.model_selection import train_test_split, GridSearchCV

# make_pipeline is a utility to create a pipeline of transforms with a final estimator, simplifying the model building process.
from sklearn.pipeline import make_pipeline

from app import model

# Load SpaCy's English model. This model has been trained on web content and includes vocabulary, syntax, entities, and word vectors.
nlp = spacy.load("en_core_web_sm")

# Initialize the Flask application. This object implements a WSGI application and acts as the central object.
app = Flask(__name__)

def preprocess_text(text):
    """
    Text preprocessing function that uses spaCy for NLP tasks like tokenization and lemmatization.
    """
    # Use regular expressions to replace URLs and email addresses in the text with placeholders.
    text = re.sub(r'https?://\S+|www\.\S+', ' urlplaceholder ', text)
    text = re.sub(r'\S*@\S*\s?', ' emailplaceholder ', text)

    # Process the text with spaCy, tokenizing it and performing lemmatization and stop word removal.
    doc = nlp(text)
    print(doc)
    clean_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(clean_tokens)

def load_and_preprocess_data(filename='data.csv'):
    """
    Load a dataset from a CSV file and preprocess the text.
    """
    # Use pandas to read the CSV file.
    dataset = pd.read_csv(filename)
    # Apply the preprocess_text function to each text in the dataset.
    dataset['text'] = dataset['text'].apply(preprocess_text)
    # Convert the 'label' column to integer type.
    dataset['label'] = dataset['label'].astype(int)
    return dataset

def train_model(dataset):
    """
    Train a logistic regression model on the preprocessed text data.
    """
    # Split the dataset into features (X) and target variable (y).
    X = dataset['text']
    y = dataset['label']
    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline that first transforms the text data into TF-IDF vectors, then applies logistic regression.
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 3)),
        LogisticRegression()
    )
    # Define parameters for GridSearchCV to tune.
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000],
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
        'logisticregression__C': [0.1, 1, 10]
    }

    # Perform grid search with cross-validation to find the best parameters.
    cv = GridSearchCV(pipeline, parameters, cv=5)
    cv.fit(X_train, y_train)
    print(f"Best parameters: {cv.best_params_}")
    return cv.best_estimator_

# Define the route for the web application's homepage.
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form.
        user_name = request.form.get('name')
        user_input = request.form.get('text')
        # Preprocess the input and make a prediction with the trained model.
        preprocessed_text = preprocess_text(user_input)
        prediction_prob = model.predict_proba([preprocessed_text])[0]
        # Calculate the probability of the text being phishing.
        phishing_prob = prediction_prob[1] * 100

        # Render the results page with the prediction.
        return render_template('results.html', name=user_name, text=user_input, probability="{:.2f}%".format(phishing_prob))
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask application.
    app.run(debug=True)
