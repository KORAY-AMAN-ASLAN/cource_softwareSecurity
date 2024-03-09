# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For model persistence

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing text function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if not word in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Load dataset
dataset = pd.read_csv('data.csv')

# Apply preprocessing to the dataset
dataset['text'] = dataset['text'].apply(preprocess_text)

# Feature Extraction with TfidfVectorizer, now including bi-grams
tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(dataset['text']).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['label'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and hyperparameter tuning
parameters = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
classifier = GridSearchCV(LogisticRegression(), parameters, cv=3)
classifier.fit(X_train, y_train)

# Best model after tuning
best_clf = classifier.best_estimator_

# Predict on the test set
predictions = best_clf.predict(X_test)

# Predict probabilities on the test set for further analysis
probabilities = best_clf.predict_proba(X_test)

# Evaluate the model
print("Best Model Parameters:", classifier.best_params_)
print("Accuracy on Test Set:", accuracy_score(y_test, predictions))
print("Classification Report on Test Set:\n", classification_report(y_test, predictions))

# Save the model and vectorizer to disk
joblib.dump(best_clf, 'phishing_detection_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Load new data (unseen)
new_data = pd.read_csv('new_emails.csv')
new_data['text'] = new_data['text'].apply(preprocess_text)

# Load the model and vectorizer from disk
best_clf = joblib.load('phishing_detection_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Transform the new data using the loaded TF-IDF vectorizer
new_data_transformed = tfidf_vectorizer.transform(new_data['text'])

# Predict probabilities on the new data
new_predictions_prob = best_clf.predict_proba(new_data_transformed)

# Create a DataFrame to display the probabilities with the corresponding texts
probabilities_df = pd.DataFrame(new_predictions_prob, columns=['Legit_Prob', 'Phishing_Prob'])
result_df = pd.concat([new_data, probabilities_df], axis=1)

print(result_df)
