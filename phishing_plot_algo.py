import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

# Example texts to classify
texts = ["Example of legitimate text", "Example of phishing text"]
# For demonstration, we'll use the same texts without actual preprocessing
# In a real scenario, ensure to preprocess these texts as you did for training
preprocessed_texts = texts  # Replace this with actual preprocessing if necessary

# Function to predict probabilities using a model
def predict_probabilities(model, texts):
    # Here, we directly use the texts assuming they're preprocessed appropriately for the model
    # Replace this with model.predict_proba(preprocessed_texts) as per your model's preprocessing requirement
    return model.predict_proba(texts)[:, 1]  # Probability of being phishing

# Predicting probabilities for each model
logistic_regression_probs = predict_probabilities(logistic_regression_model, preprocessed_texts)
random_forest_probs = predict_probabilities(random_forest_model, preprocessed_texts)
gradient_boosting_probs = predict_probabilities(gradient_boosting_model, preprocessed_texts)

# Data for plotting
models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
probabilities = [logistic_regression_probs[1], random_forest_probs[1], gradient_boosting_probs[1]]  # Using phishing text probabilities

# Plotting the results
fig, ax = plt.subplots()
y_pos = np.arange(len(models))
ax.barh(y_pos, probabilities, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.invert_yaxis()  # Highest probability at the top
ax.set_xlabel('Probability')
ax.set_title('Phishing Text Classification Probability by Model')

plt.show()