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

# Updated example texts for classification
# Updated example texts for classification
# Updated example texts for classification
texts = [
    # Legitimate Emails
    "Hello [Name], Just a reminder that your next appointment is scheduled for [Date]. Please confirm your attendance. Regards, [Your Doctor's Office]",
    "Good day! Your membership renewal was successful. For details on membership benefits, visit our site. Best, [Membership Team]",
    "Greetings from [Your Library]! The book you reserved is now available for pickup until [Date]. Enjoy reading!",
    "Your feedback is important to us! Please let us know about your experience with our service by filling out this quick survey. Thank you, [Customer Service Team]",

    # Phishing Emails
    "Warning: Your subscription will be cancelled unless you update your billing info now at [Suspicious Link]. Don’t miss out!",
    "You’ve won an iPhone! Click [Suspicious Link] to claim your prize now. Only a few left!",
    "Security Alert: Your account was accessed from an unknown device. Secure it right away here [Suspicious Link].",
    "Invoice #4563 is overdue. Immediate payment required to avoid service termination. Pay now: [Suspicious Link]"
]



# Function to predict probabilities using a model and convert to percent
def predict_probabilities(model, texts):
    # Assuming the models are pipelines that can directly handle raw text input
    # Multiplies by 100 to convert probabilities to percentages
    return [prob[1] * 100 for prob in model.predict_proba(texts)]  # Probability of being phishing in percent

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

# Adjust the subplot grid size based on the number of texts
num_texts = len(texts)
cols = 2
rows = num_texts // cols + (num_texts % cols > 0)

# Create a figure and a grid of subplots adjusted for the number of texts
fig, axs = plt.subplots(rows, cols, figsize=(12, 8), constrained_layout=True)

# Make sure axs is a 2D array for easy iteration even if there's only one row
if num_texts <= cols:
    axs = [axs]

# Loop through each type of message
for idx, ax in enumerate(np.array(axs).flat):
    if idx >= len(texts):  # If there are more subplots than texts, break the loop
        break
    probabilities = [
        logistic_regression_probs[idx],
        random_forest_probs[idx],
        gradient_boosting_probs[idx]
    ]
    # Plot each model's prediction with a distinct color and label
    for model_idx, model in enumerate(models):
        ax.bar(model, probabilities[model_idx], color=colors[model_idx], label=model, edgecolor='black')
    ax.set_title(f'Email {idx + 1} - {email_type} Likelihood')
    ax.set_ylabel('Phishing Likelihood (%)')
    ax.set_ylim(0, 100)  # Adjusting probability range to percentage

# Handle scenario where there are less emails than subplots
for idx in range(len(texts), len(np.array(axs).flat)):
    fig.delaxes(np.array(axs).flat[idx])

fig.legend(models, loc='upper right')
plt.show()

