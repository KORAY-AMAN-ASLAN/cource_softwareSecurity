# Email Classification Web Application using Flask and Machine Learning

This Python script leverages Flask to create a web application that classifies email texts as either legitimate or phishing. It integrates natural language processing (NLP), machine learning (ML), and web development to offer an interactive platform for email classification.

![Flask Form Page](URL_TO_YOUR_FLASK_FORM_IMAGE)
*The Flask-based web application interface for email classification.*

## Key Components and Workflow

- **Library Imports**: Imports essential libraries for web development, data manipulation, NLP, machine learning, and optional visualization.

- **Model Loading**: Loads pre-trained models (Logistic Regression, Random Forest, Gradient Boosting) for classification.

- **Email Text Preprocessing**: Defines a function `preprocess_text` for cleaning and normalizing input email texts.

- **Data Loading and Preprocessing for Training**: Automates loading and preprocessing of new training data.

- **Model Training with Hyperparameter Optimization**: `train_model` function for training and optimizing models.

- **Flask Web Application Setup**: Sets up a Flask web application to handle requests and serve predictions.

![Phishing Email Attempt](URL_TO_YOUR_PHISHING_EMAIL_ATTEMPT_IMAGE)
*Example of a phishing email attempt identified by the application.*

![Legitimate Email](URL_TO_YOUR_LEGITIMATE_EMAIL_IMAGE)
*Example of the application successfully identifying a legitimate email.*

## Model Comparison

The application utilizes various machine learning models to classify emails. Below is a graph comparing their performances:

![Model Comparison Graph](URL_TO_YOUR_MODEL_COMPARISON_GRAPH_IMAGE)
*Comparison of machine learning models based on their ability to classify emails.*

## Getting Started

To run this project locally, clone the repository and install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
