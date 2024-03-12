"""
The script employs machine learning (ML) models to classify email texts into 'Legitimate' or 'Phishing' categories,
 focusing on the probability of an email being phishing. Utilizing Logistic Regression, Random Forest,
 and Gradient Boosting models, it predicts and visualizes the likelihood of phishing for given email examples.
  This process illustrates how ML can be applied to cybersecurity,
  specifically in automating the detection and analysis of phishing threats.

Workflow:
1. Load Pre-trained ML Models:
The models, trained on labeled datasets distinguishing phishing from legitimate emails,
are loaded to evaluate new texts.

2. Classify Example Emails:
Two emails are classified—one simulating a legitimate service provider's message and the other mimicking a phishing attempt.
These examples help demonstrate the models' ability to assess and differentiate email types based on learned patterns.

3. Predict and Convert Probabilities:
For each text, the models predict the probability of being phishing.
These predictions are converted to percentages to simplify interpretation,
aiming to reflect realistic scenarios where phishing emails might show high probabilities (e.g., 97%),
 while legitimate ones show significantly lower probabilities (e.g., 33% or 55%).

4. Terminal Output and Visualization:
Classification probabilities for each email and model are printed to the terminal,
providing immediate insight into each model's evaluation. Furthermore,
the script generates bar plots to visually compare the models' assessments, enhancing the comparative analysis.

5. Insightful Comparative Analysis:
By analyzing the output and visualizations, users can observe how different models rate the same texts,
offering valuable perspectives on model performance, reliability, and suitability for phishing detection tasks.

This approach not only highlights the practical application of ML in identifying phishing attempts but also showcases,
 the nuanced capabilities of different models, contributing to the development of effective, AI-driven solutions for email security.
"""



import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

# Example texts to classify
texts = [
    # Legitimate Emails
    "Dear customer, We noticed unusual activity in your account and need you to verify your identity. Please visit our official website and log in to your account for verification.",
    "Hello, Thank you for your recent purchase. We hope you enjoyed your shopping experience. For any queries, feel free to contact our support team.",

    # Phishing Emails
    "URGENT: Your account has been compromised! Click this link immediately to verify your account: http://phishingsite.com/login.",
    "Your account has been flagged for suspicious activity. Please verify your information to avoid suspension: [Malicious Link]",
]

import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

# Example texts to classify
texts = [
    # Legitimate Emails
    "Hi Adam, Hope you're having a great week! Just wanted to check in and see if you had any questions about the new project. Best, Larry Page",
    "Hello, Thank you for your recent purchase. We hope you enjoyed your shopping experience. For any queries, feel free to contact our support team.",

    # Phishing Emails
    "URGENT: Your account has been compromised! Click this link immediately to verify your account: http://phishingsite.com/login.",
    "Your account has been flagged for suspicious activity. Please verify your information to avoid suspension: [Malicious Link]",
]


# Function to predict probabilities using a model and convert to percent
def predict_probabilities(model, texts):
    # Multiplies by 100 to convert probabilities to percentages
    return model.predict_proba(texts)[:, 1] * 100  # Probability of being phishing in percent


# Predicting probabilities for each model
logistic_regression_probs = predict_probabilities(logistic_regression_model, texts)
random_forest_probs = predict_probabilities(random_forest_model, texts)
gradient_boosting_probs = predict_probabilities(gradient_boosting_model, texts)

# Print adjusted results in terminal
print("Phishing Classification Probabilities (%):")
for idx, email in enumerate(texts):
    email_type = 'Legitimate' if idx < len(texts) // 2 else 'Phishing'
    print(f"\nEmail {idx + 1} ({email_type}): \"{email}\"")
    print(f"  Logistic Regression - Phishing Likelihood: {logistic_regression_probs[idx]:.2f}%")
    print(f"  Random Forest - Phishing Likelihood: {random_forest_probs[idx]:.2f}%")
    print(f"  Gradient Boosting - Phishing Likelihood: {gradient_boosting_probs[idx]:.2f}%")

# Data for plotting
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
colors = ['blue', 'green', 'red']  # Different color for each model

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Loop through each type of message
for idx, ax in enumerate(axs.flat):
    probabilities = [
        logistic_regression_probs[idx],
        random_forest_probs[idx],
        gradient_boosting_probs[idx]
    ]

    ax.bar(models, probabilities, color=colors)
    ax.set_title(f'Email {idx + 1} - Phishing Classification Probabilities')
    ax.set_ylabel('Phishing Likelihood (%)')
    ax.set_ylim(0, 100)  # Adjusting probability range to percentage

# Add a legend outside the rightmost subplot
fig.legend(models, loc='upper right')

plt.tight_layout()
plt.show()



