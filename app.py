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

# Initialize Flask app
app = Flask(__name__)


# Function to preprocess text
def preprocess_text(text):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if
                    token.isalpha() and token not in stopwords.words('english')]
    return " ".join(clean_tokens)


# Load and preprocess dataset
def load_and_preprocess_data():
    dataset = pd.read_csv('data.csv')
    dataset['text'] = dataset['text'].apply(preprocess_text)
    return dataset


# Train model
def train_model(dataset):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['label'], test_size=0.2,
                                                        random_state=42)

    # Create a pipeline with TF-IDF Vectorizer and Logistic Regression
    model = make_pipeline(TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), LogisticRegression())
    model.fit(X_train, y_train)
    return model


# Load dataset and train model on startup
dataset = load_and_preprocess_data()
model = train_model(dataset)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form['name']
        user_input = request.form['text']

        # Preprocess and predict
        preprocessed_text = preprocess_text(user_input)
        prediction_prob = model.predict_proba([preprocessed_text])[0]

        # Assuming the second class is "phishing"
        phishing_prob = prediction_prob[1] * 100

        # Render the results page with the user's name, the text they submitted, and the prediction probability
        return render_template('results.html', name=user_name, text=user_input,
                               probability="{:.2f}%".format(phishing_prob))

    # For a GET request, just show the index page
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)