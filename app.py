# Importing necessary libraries and modules
from flask import Flask, render_template, jsonify, request  # For web app development and handling HTTP requests
import pandas as pd  # For data manipulation and analysis
import re  # For regular expression operations, useful in text processing
import spacy  # For advanced natural language processing tasks
# sklearn imports for machine learning functionality
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical data
from sklearn.linear_model import LogisticRegression  # Machine learning model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Additional machine learning models
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and optimizing model parameters
from sklearn.pipeline import make_pipeline  # For creating a processing/modeling pipeline

# Load the spaCy model for English, used for various NLP tasks like tokenization and lemmatization
nlp = spacy.load("en_core_web_sm")

# Initialize the Flask application
app = Flask(__name__)

# Define function for preprocessing text input
def preprocess_text(text):
    # Replace URLs and email addresses with placeholders
    text = re.sub(r'https?://\S+|www\.\S+', ' urlplaceholder ', text)
    text = re.sub(r'\S*@\S*\s?', ' emailplaceholder ', text)
    # Use spaCy for further processing like lemmatization and stop word removal
    doc = nlp(text)
    clean_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(clean_tokens)  # Return preprocessed text as a single string

# Function to load data, apply preprocessing, and prepare it for model training or predictions
def load_and_preprocess_data(filename='data.csv'):
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)
    return dataset

# Function to train a machine learning model based on preprocessed text data
def train_model(dataset, model_type='logistic_regression'):
    # Split data into features and target variable
    X = dataset['text']
    y = dataset['label']
    # Further split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Model selection based on input parameter
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100)
    # Create a pipeline with TF-IDF vectorization and the selected model
    pipeline = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 3)), model)
    # Parameters for optimizing the pipeline
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000],
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
    }
    # If logistic regression, add specific parameters for optimization
    if model_type == 'logistic_regression':
        parameters['logisticregression__C'] = [0.1, 1, 10]
    # Use GridSearchCV for parameter optimization
    cv = GridSearchCV(pipeline, parameters, cv=5)
    cv.fit(X_train, y_train)
    print(f"Best parameters: {cv.best_params_}")
    return cv.best_estimator_  # Return the trained model

# Define the route for handling web requests to the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle JSON data for AJAX requests
        if request.headers.get('Content-Type') == 'application/json':
            data = request.get_json()
            subject = data['subject']
            message = data['message']
            combined_input = f"Subject: {subject}\nMessage: {message}"
            preprocessed_text = preprocess_text(combined_input)
            # Assume model is already loaded; predict phishing probability
            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100
            return jsonify(probability="{:.2f}".format(phishing_prob))
        else:
            # Handle form data for regular form submissions
            user_input = request.form.get('text')
            preprocessed_text = preprocess_text(user_input)
            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100
            return render_template('results.html', probability="{:.2f}%".format(phishing_prob))
    return render_template('results.html')  # Render the homepage with the form

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
