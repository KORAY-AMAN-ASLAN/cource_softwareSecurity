from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and not token in stop_words]
    return " ".join(clean_tokens)

# Load and preprocess the dataset
def load_and_preprocess_data(filename='data.csv'):
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)  # Convert labels to integers
    return dataset

# Train the model using the dataset
def train_model(dataset):
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use TfidfVectorizer within a Pipeline
    model = make_pipeline(TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), LogisticRegression())
    model.fit(X_train, y_train)
    return model

# Load dataset and train the model
dataset = load_and_preprocess_data('data.csv')  # Make sure to point to the correct file
model = train_model(dataset)

# Flask route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form.get('name')
        user_input = request.form.get('text')
        preprocessed_text = preprocess_text(user_input)
        prediction_prob = model.predict_proba([preprocessed_text])[0]
        phishing_prob = prediction_prob[1] * 100  # Probability of being phishing

        return render_template('results.html', name=user_name, text=user_input, probability="{:.2f}%".format(phishing_prob))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
