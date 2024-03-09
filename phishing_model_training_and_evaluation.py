import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Download necessary NLTK datasets
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocess text function
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and not token in stop_words]
    return " ".join(clean_tokens)

# Enhanced function to load and preprocess dataset
def load_and_preprocess_data(filename='data.csv'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} was not found.")
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)  # Ensure labels are integers
    return dataset

# Function to train and return a pipeline model
def train_model(dataset):
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
        ('clf', LogisticRegression())
    ])

    parameters = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__solver': ['lbfgs', 'liblinear']
    }
    grid_search = GridSearchCV(pipeline, parameters, cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate the model with cross-validation
    cv_scores = cross_val_score(grid_search, X, y, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores)}")

    # Final evaluation on the test set
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    print("Accuracy on Test Set:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report on Test Set:\n", classification_report(y_test, predictions))
    try:
        roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        print(f"ROC-AUC Score: {roc_auc}")
    except AttributeError:
        # Some models might not support predict_proba
        print("ROC-AUC Score not available for this model configuration.")

    return best_model

try:
    # Load dataset and train the model
    dataset = load_and_preprocess_data('data.csv')  # Adjust filename as necessary
    model = train_model(dataset)

    # Save the trained model to disk
    joblib.dump(model, 'phishing_detection_enhanced_pipeline.joblib')
    print("Model saved successfully.")

    # Example of loading the model (uncomment to use)
    # loaded_model = joblib.load('phishing_detection_enhanced_pipeline.joblib')
except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
