from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)


def preprocess_text(text):
    """
    Enhanced text preprocessing that includes URL and email handling,
    along with stop words removal, tokenization, and lemmatization.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Handling URLs and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', ' urlplaceholder ', text)
    text = re.sub(r'\S*@\S*\s?', ' emailplaceholder ', text)

    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and not token in stop_words]
    return " ".join(clean_tokens)


def load_and_preprocess_data(filename='data.csv'):
    """
    Load and preprocess the dataset from a CSV file.
    """
    dataset = pd.read_csv(filename)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    dataset['label'] = dataset['label'].astype(int)
    return dataset


def train_model(dataset):
    """
    Train the model using the preprocessed dataset, allowing for hyperparameter tuning and model selection.
    """
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(
        TfidfVectorizer(max_features=5000, ngram_range=(1, 3)),
        LogisticRegression()
    )
    parameters = {
        'tfidfvectorizer__max_features': [3000, 5000],
        'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
        'logisticregression__C': [0.1, 1, 10]
    }

    cv = GridSearchCV(pipeline, parameters, cv=5)
    cv.fit(X_train, y_train)
    print(f"Best parameters: {cv.best_params_}")
    return cv.best_estimator_


# Load dataset and train the model
dataset = load_and_preprocess_data('data.csv')
model = train_model(dataset)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Flask route for the home page, handling both GET and POST requests.
    """
    if request.method == 'POST':
        user_name = request.form.get('name')
        user_input = request.form.get('text')
        preprocessed_text = preprocess_text(user_input)
        prediction_prob = model.predict_proba([preprocessed_text])[0]
        phishing_prob = prediction_prob[1] * 100  # Probability of being phishing

        return render_template('results.html', name=user_name, text=user_input,
                               probability="{:.2f}%".format(phishing_prob))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
