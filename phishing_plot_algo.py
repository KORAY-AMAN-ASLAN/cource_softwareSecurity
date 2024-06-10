"""


  Authour:  Koray Aman Arabzadeh
 Cource: Mjukvarusäkerhet, Software Security ,  Mittuniversitetet
 Year: 2024-04-01

 Video can be found here: https://www.youtube.com/watch?v=-3TfoUi6oTk


"""



import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

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



def predict_probabilities(model, texts):

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

num_texts = len(texts)
cols = 2
rows = num_texts // cols + (num_texts % cols > 0)

fig, axs = plt.subplots(rows, cols, figsize=(12, 8), constrained_layout=True)

if num_texts <= cols:
    axs = [axs]

# Loop through each type of message
for idx, ax in enumerate(np.array(axs).flat):
    if idx >= len(texts):  
        break
    probabilities = [
        logistic_regression_probs[idx],
        random_forest_probs[idx],
        gradient_boosting_probs[idx]
    ]
    for model_idx, model in enumerate(models):
        ax.bar(model, probabilities[model_idx], color=colors[model_idx], label=model, edgecolor='black')
    ax.set_title(f'Email {idx + 1} - {email_type} Likelihood')
    ax.set_ylabel('Phishing Likelihood (%)')
    ax.set_ylim(0, 100)  # Adjusting probability range to percentage

for idx in range(len(texts), len(np.array(axs).flat)):
    fig.delaxes(np.array(axs).flat[idx])

fig.legend(models, loc='upper right')
plt.show()

