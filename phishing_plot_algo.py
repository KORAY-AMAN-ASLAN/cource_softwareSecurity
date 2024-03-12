import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load data from CSV file
data = pd.read_csv('data.csv')  # Adjust the filename as per your CSV file

# Extract texts and labels
texts = data['text'].tolist()
y_true = data['label'].tolist()

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, y_true, test_size=0.5, random_state=42)

# Define a function to create a pipeline and train the model
def create_and_train_pipeline(X_train, y_train, classifier):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', classifier),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Initialize and train models (same as before)
models = {
    "Logistic Regression": LogisticRegression()
   # "Random Forest": RandomForestClassifier(),
   # "Gradient Boosting": GradientBoostingClassifier()
}

trained_models = {name: create_and_train_pipeline(X_train, y_train, model) for name, model in models.items()}

# Predict probabilities for the test set (same as before)
predictions = {name: model.predict_proba(X_test)[:, 1] for name, model in trained_models.items()}

# Function to plot predictions (same as before)
def plot_predictions(texts, predictions, y_true):
    plt.figure(figsize=(14, 8))
    x = np.arange(len(texts))  # the label locations
    width = 0.25  # the width of the bars

    legit_indices = [i for i, label in enumerate(y_true) if label == 0]
    phishing_indices = [i for i, label in enumerate(y_true) if label == 1]

    fig, ax = plt.subplots()
    for i, (model_name, model_predictions) in enumerate(predictions.items()):
        ax.bar(x[legit_indices] + i*width, model_predictions[legit_indices], width, label=f"{model_name} (Legit)")
        ax.bar(x[phishing_indices] + i*width, model_predictions[phishing_indices], width, label=f"{model_name} (Phishing)")

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Probabilities')
    ax.set_title('Phishing Probability by Model and Text')
    ax.set_xticks(x + width / len(predictions) - width/2)
    ax.set_xticklabels(texts, rotation=45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.show()

# Plotting predictions for the test set texts (same as before)
plot_predictions(X_test, predictions, y_test)
