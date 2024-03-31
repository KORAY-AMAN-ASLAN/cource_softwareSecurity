# Email Phishing Classification with Flask Web Application

This Python script is designed to create a web application using Flask that classifies email text as either legitimate or phishing. It utilizes machine learning models that have been pre-trained to identify characteristics of phishing attempts within email content. The script demonstrates how to load these models, use them for classification, and then serve the results through a web interface. It is an integration of natural language processing (NLP), machine learning (ML), and web development to provide an interactive platform for email classification.

## Key Components and Workflow:

1. **Library Imports**: The script begins by importing necessary libraries for web development (Flask), data manipulation (Pandas), NLP (spaCy), machine learning (Scikit-learn, joblib for model loading), and visualization (matplotlib for optional plotting).

2. **Model Loading**: Pre-trained models for Logistic Regression, Random Forest, and Gradient Boosting are loaded from disk. These models are assumed to be trained on a dataset where email texts are labeled as 'legitimate' or 'phishing', learning from features extracted via NLP techniques.

3. **Email Text Preprocessing**: A function 'preprocess_text' is defined to clean and normalize the input email texts, by removing or replacing URLs and email addresses, optionally filtering out non-alphanumeric characters and stopwords, and normalizing the text case. This preprocessing mimics the treatment of data during the model training phase and is crucial for making accurate predictions.

4. **Data Loading and Preprocessing for Training**: Another function 'load_and_preprocess_data' automates the loading of new training data from a CSV file, applying the preprocessing steps to the 'text' column, and ensuring the 'label' column is in the correct format. This functionality is key for retraining models or evaluating their performance on new data.

5. **Model Training with Hyperparameter Optimization**: The 'train_model' function orchestrates the training of a specified ML model on the preprocessed dataset, employing GridSearchCV for hyperparameter tuning to optimize model performance. This step is vital for fine-tuning models based on new data or different classification tasks.

6. **Flask Web Application Setup**: Utilizing Flask, the script sets up a web application that can handle GET and POST requests. GET requests simply render the homepage, while POST requests accept submitted email texts, preprocess them, classify them using the loaded ML models, and return the classification probabilities as phishing or legitimate.

7. **Results Visualization and Output**: Although primarily focused on serving predictions through a web interface, the script includes commented sections that suggest how one might extend it to visualize classification results using matplotlib.

8. **Execution and Model Management**: In the script's main block, there's functionality for loading a dataset, training a model, and saving it to disk. This illustrates how the web application's underlying ML model can be updated or replaced without modifying the core application logic.

Overall, this script exemplifies a practical application of combining NLP and ML for email classification within a web-based interface, showcasing a pipeline from model training to real-time prediction serving.

## Additional Details:

- **Author**: Koray Aman Arabzadeh
- **Course**: Mjukvarusäkerhet, Software Security, Mittuniversitetet
- **Year**: 2024-04-01

- **Video**: [Link to Video](https://www.youtube.com/watch?v=-3TfoUi6oTk)

## Email Classification and Model Comparison

### Legitimate vs. Phishing Email Classification

The script employs machine learning (ML) models to classify email texts into 'Legitimate' or 'Phishing' categories, focusing on the probability of an email being phishing. Utilizing Logistic Regression, Random Forest, and Gradient Boosting models, it predicts and visualizes the likelihood of phishing for given email examples. This process illustrates how ML can be applied to cybersecurity, specifically in automating the detection and analysis of phishing threats.

#### Workflow:

1. **Load Pre-trained ML Models**: The models, trained on labeled datasets distinguishing phishing from legitimate emails, are loaded to evaluate new texts.

2. **Classify Example Emails**: Example emails are classified into 'Legitimate' and 'Phishing' categories to demonstrate the models' ability to assess and differentiate email types based on learned patterns.

3. **Predict and Convert Probabilities**: For each text, the models predict the probability of being phishing, converted to percentages for interpretation.

4. **Terminal Output and Visualization**: Classification probabilities for each email and model are printed to the terminal, providing immediate insight into each model's evaluation. Additionally, the script generates bar plots to visually compare the models' assessments.

5. **Insightful Comparative Analysis**: Users can analyze the output and visualizations to observe how different models rate the same texts, offering valuable perspectives on model performance, reliability, and suitability for phishing detection tasks.

This approach highlights the practical application of ML in identifying phishing attempts and showcases the nuanced capabilities of different models, contributing to the development of effective, AI-driven solutions for email security.

#### Additional Details:

- **Author**: Koray Aman Arabzadeh
- **Course**: Mjukvarusäkerhet, Software Security, Mittuniversitetet
- **Year**: 2024-04-01
- **Video**: [Link to Video](https://www.youtube.com/watch?v=-3TfoUi6oTk)

## Instructions for Running the Script:

To run the Flask web application and classify email texts:

1. Clone the repository containing the script and models.
2. Ensure you have Python installed on your system.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Run the script using `python script_name.py`.
5. Access the web application through the provided URL in the terminal or by navigating to `http://localhost:5000` in your web browser.
6. Submit email texts through the provided form to classify them as either legitimate or phishing.

