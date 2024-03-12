"""
This Python script is designed to create a web application using Flask that classifies email text as either legitimate,
 or phishing.
It utilizes machine learning models that have been pre-trained to identify characteristics of phishing attempts within,
email content.
The script demonstrates how to load these models, use them for classification, and then serve the results through a web,
 interface. It is an integration of natural language processing (NLP), machine learning (ML),
and web development to provide an interactive platform for email classification.



Key Components and Workflow:

1. Library Imports:
The script begins by importing necessary libraries for web development (Flask),
data manipulation (Pandas), NLP (spaCy), machine learning (Scikit-learn, joblib for model loading),
and visualization (matplotlib for optional plotting).

2. Model Loading:
Pre-trained models for Logistic Regression,
Random Forest, and Gradient Boosting are loaded from disk.
These models are assumed to be trained on a dataset where email texts are labeled as 'legitimate' or 'phishing',
learning from features extracted via NLP techniques.

3. Email Text Preprocessing:
A function 'preprocess_text' is defined to clean and normalize the input email texts,
by removing or replacing URLs and email addresses, optionally filtering out non-alphanumeric characters and stopwords,
and normalizing the text case.
This preprocessing mimics the treatment of data during the model training phase and is crucial for making accurate predictions.

4. Data Loading and Preprocessing for Training:
Another function 'load_and_preprocess_data' automates the loading of new training data from a CSV file,
 applying the preprocessing steps to the 'text' column, and ensuring the 'label' column is in the correct format.
  This functionality is key for retraining models or evaluating their performance on new data.

5. Model Training with Hyperparameter Optimization:
The 'train_model' function orchestrates the training of a specified ML model on the preprocessed dataset,
employing GridSearchCV for hyperparameter tuning to optimize model performance.
This step is vital for fine-tuning models based on new data or different classification tasks.

6. Flask Web Application Setup:
Utilizing Flask, the script sets up a web application that can handle GET and POST requests.
GET requests simply render the homepage, while POST requests accept submitted email texts, preprocess them,
classify them using the loaded ML models, and return the classification probabilities as phishing or legitimate.

7. Results Visualization and Output:
Although primarily focused on serving predictions through a web interface,
the script includes commented sections that suggest how one might extend it to visualize classification results using matplotlib.

8. Execution and Model Management:
In the script's main block, there's functionality for loading a dataset, training a model, and saving it to disk.
This illustrates how the web application's underlying ML model,
can be updated or replaced without modifying the core application logic.

Overall, this script exemplifies a practical application of combining NLP and ML for email classification,
 within a web-based interface, showcasing a pipeline from model training to real-time prediction serving.


 Authour:
 Koray Aman Arabzadeh
 Cource: Mjukvarusäkerhet Mittuniversitetet
 Year: 2024-03-12

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

    # Process the text with spaCy: tokenize, lemmatize, and apply the specified preprocessing steps.
    doc = nlp(text)
    clean_tokens = [token.lemma_.lower() if lower_case else token.lemma_ for token in doc if
                    (not remove_stopwords or not token.is_stop) and (not remove_non_alpha or token.is_alpha)]

    return " ".join(clean_tokens)


# Load data from a CSV file, apply preprocessing, and prepare it for model training.
def load_and_preprocess_data(filename='data.csv'):
    """
    Loads phishing data from a CSV, applies text preprocessing, and ensures the label column is integer-typed.
    """
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)
    return dataset


# Train a machine learning model using the preprocessed dataset.
def train_model(dataset, model_type='logistic_regression'):
    """
    Trains a specified machine learning model on preprocessed text data, performing hyperparameter optimization.
    """
    # Split the dataset into training and testing sets.
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a dictionary of available models.
    models = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'svm': SVC(probability=True)  # Ensure probability estimation is enabled for SVC.
    }

    # Validate the selected model type and prepare the modeling pipeline with TfidfVectorizer.
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}.")
    model = models[model_type]
    pipeline = make_pipeline(TfidfVectorizer(), model)

    # Set hyperparameters for GridSearchCV.
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000, 7000],
        # Limits the number of features (unique words) to consider, reducing dimensionality and potentially improving model performance and training time.
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)]
        # Defines the range of n-values for different n-grams to be extracted. (1, 2) means using unigrams and bigrams, while (1, 3) includes unigrams, bigrams, and trigrams, capturing more context.
    }
    model_params = {
        'logistic_regression': {'logisticregression__C': [0.1, 1, 10]},
        # C is the inverse of regularization strength; smaller values specify stronger regularization. Regularization can help prevent overfitting by penalizing large coefficients.
        'random_forest': {'randomforestclassifier__n_estimators': [100, 200]},
        # n_estimators is the number of trees in the forest. More trees can lead to more robust models but increases computation.
        'gradient_boosting': {'gradientboostingclassifier__n_estimators': [100, 200]},
        # Similarly, for gradient boosting, it specifies the number of boosting stages to be run. More stages can improve accuracy but may lead to overfitting.
        'svm': {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
        # For SVM, C again controls the strength of regularization. The kernel parameter selects the type of hyperplane used to separate the data. 'linear' is for linear hyperplanes, while 'rbf' allows for non-linear boundaries.
    }

    parameters.update(model_params[model_type])

    # Optimize model parameters.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')
        cv.fit(X_train, y_train)

    # Output best parameters and model performance on test set.
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

            # Return the prediction in the appropriate format based on the request type.
            return jsonify(probability="{:.2f}".format(phishing_prob)) if request.is_json else render_template(
                'index.html', probability="{:.2f}%".format(phishing_prob))
        except Exception as e:
            # Handle unexpected errors gracefully.
            return jsonify(error=str(e)), 500

    # Render the homepage for GET requests.
    return render_template('index.html')


if __name__ == '__main__':
    # Train and save the model, or load an existing model, then run the Flask app.
    # This section can be commented out or modified based on the deployment strategy to avoid retraining on every launch.
   # dataset = load_and_preprocess_data('data.csv')
    # logistic_regression_model.pkl
    # mrandom_forest_model.pkl
    # gradient_boosting_model.pkl

    #model = train_model(dataset, 'logistic_regression')
   # joblib.dump(model, MODEL_PATH)
    app.run(debug=True)