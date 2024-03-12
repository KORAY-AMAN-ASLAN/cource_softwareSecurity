# Importing necessary libraries and modules
from flask import Flask, render_template, jsonify, \
    request  # Flask for web app development, render_template for rendering HTML, request for handling requests
import pandas as pd  # pandas for data manipulation and analysis
import re  # re for regular expression operations
import spacy  # spaCy for advanced NLP tasks
# Importing machine learning models and utilities from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF transformer for text vectorization
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble models
from sklearn.model_selection import train_test_split, \
    GridSearchCV  # Utilities for splitting data and hyperparameter tuning
from sklearn.pipeline import make_pipeline  # Utility to create a pipeline of transforms with a final estimator
import matplotlib.pyplot as plt

from phishing_model_training_and_evaluation import model

# Loading the spaCy English language model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Initializing the Flask application
app = Flask(__name__)


def preprocess_text(text):
    """
    Preprocesses input text for NLP tasks.
    - Uses regular expressions to replace URLs and email addresses with placeholders.
    - Utilizes spaCy for tokenization, lemmatization, and removal of stop words.
    - Converts tokens to lowercase.
    """
    # Replacing URLs with a placeholder
    text = re.sub(r'https?://\S+|www\.\S+', ' urlplaceholder ', text)
    # Replacing email addresses with a placeholder
    text = re.sub(r'\S*@\S*\s?', ' emailplaceholder ', text)

    # Tokenizing and processing the text using spaCy
    doc = nlp(text)
    # Building a list of lemmatized, lowercase tokens that are not stop words and are alphabetic
    clean_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    # Joining processed tokens back into a single string
    return " ".join(clean_tokens)


def load_and_preprocess_data(filename='data.csv'):
    """
    Loads and preprocesses dataset from a CSV file.
    - Reads data into a pandas DataFrame.
    - Applies text preprocessing to the 'text' column.
    - Converts the 'label' column to integer type.
    """
    # Reading the dataset from a CSV file into a pandas DataFrame
    dataset = pd.read_csv(filename)
    # Applying the text preprocessing function to each item in the 'text' column
    dataset['text'] = dataset['text'].apply(preprocess_text)
    # Converting the 'label' column to integer type for model training
    dataset['label'] = dataset['label'].astype(int)
    return dataset


def train_model(dataset, model_type='gradient_boosting'):
    """
    Trains a machine learning model on the preprocessed dataset.
    - Splits the dataset into training and testing sets.
    - Depending on the model_type parameter, selects a Logistic Regression, Random Forest, or Gradient Boosting model.
    - Constructs a pipeline with TF-IDF vectorization and the chosen model.
    - Uses GridSearchCV for hyperparameter tuning.
    - Returns the trained model with the best parameters found.
    """

    # Splitting the dataset into features (X) and target variable (y)
    X = dataset['text']
    y = dataset['label']
    # Further splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Selecting the model based on model_type
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100)

    # Creating a pipeline with TF-IDF vectorization and the selected model
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 3)),
        model
    )

    # Defining hyperparameters for GridSearchCV
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000],
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
    }

    # Adding model-specific hyperparameters
    if model_type == 'logistic_regression':
        parameters['logisticregression__C'] = [0.1, 1, 10]

    # Using GridSearchCV for hyperparameter tuning with cross-validation
    cv = GridSearchCV(pipeline, parameters, cv=5)
    # Fitting the model to the training data
    cv.fit(X_train, y_train)
    # Printing the best parameters found
    print(f"Best parameters: {cv.best_params_}")
    # Returning the best model found
    return cv.best_estimator_





@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Defines the behavior of the homepage.
    - On POST request (form submission), preprocesses the user input, makes a prediction, and renders the results.
    - On GET request, renders the homepage with the input form.
    """

    if request.method == 'POST':
        # Check if the request is AJAX
        if request.headers.get('Content-Type') == 'application/json':
            # Parsing JSON data from the request
            data = request.get_json()
            subject = data['subject']
            message = data['message']

            # Combine subject and message or process them as needed for your model
            combined_input = f"Subject: {subject}\nMessage: {message}"
            preprocessed_text = preprocess_text(combined_input)
            prediction_prob = model.predict_proba([preprocessed_text])[0]
            phishing_prob = prediction_prob[1] * 100

            # Return JSON response containing the probability
            return jsonify(probability="{:.2f}".format(phishing_prob))

        # Fallback for non-AJAX POST requests, if necessary
        user_input = request.form.get('text')
        preprocessed_text = preprocess_text(user_input)
        prediction_prob = model.predict_proba([preprocessed_text])[0]
        phishing_prob = prediction_prob[1] * 100
        return render_template('index.html', probability="{:.2f}%".format(phishing_prob))

    # Rendering the homepage with the input form for GET requests
    return render_template('index.html')


# Your app initialization and model training here...

if __name__ == '__main__':
    app.run(debug=True)
