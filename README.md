# Cybersecurity Project: NLP and ML for Email Phishing Detection

This project, developed as part of a university cybersecurity course, showcases a Flask web application for classifying email texts. It uses advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques to differentiate between legitimate and phishing emails. The application demonstrates the practical application of these technologies in identifying phishing attempts, highlighting the synergy between NLP, ML, and web development.

## Project Overview and Workflow:

1. **Library Imports**: The project starts by importing critical libraries required for web development with Flask, data handling with Pandas, NLP processing with spaCy, ML modeling with Scikit-learn, and optional data visualization with matplotlib.

2. **Model Loading**: We use pre-trained ML models like Logistic Regression, Random Forest, and Gradient Boosting. These models have been trained on datasets categorizing emails into 'legitimate' and 'phishing', showcasing how NLP techniques can extract meaningful features for classification.

3. **Email Text Preprocessing**: The `preprocess_text` function is designed to clean and normalize email texts, crucial for maintaining consistency with the data used to train the models. This step involves removing or replacing URLs, email addresses, and optionally filtering out non-alphanumeric characters and stopwords, along with normalizing text case.

4. **Data Handling for Training**: The `load_and_preprocess_data` function simplifies the process of incorporating new training data from CSV files, applying necessary preprocessing to ensure data is correctly formatted for model training and evaluation.

5. **Model Training and Optimization**: Through the `train_model` function, the project not only trains specified ML models on preprocessed datasets but also employs techniques like GridSearchCV for hyperparameter optimization, enhancing model accuracy and performance.

6. **Flask Web Application Deployment**: The application, built with Flask, is designed to process both GET and POST requests, allowing users to interactively submit email texts for classification and receive predictions on whether the email is phishing or legitimate.

7. **Visualization and Results Interpretation**: Although the primary focus is on classification, the project includes provisions for extending functionality to visualize results with matplotlib, offering insights into model predictions and performance.

8. **Application and Model Management**: Demonstrating end-to-end application functionality, from data loading and model training to deployment and user interaction, without needing to alter the core application for model updates or replacements.

## Project Insights:

This cybersecurity course project illustrates the potent combination of NLP and ML in a real-world application, emphasizing the critical role of these technologies in enhancing cybersecurity measures against phishing threats.

### Course Details:

- **Project for**: Cybersecurity Course
- **University**: [University Name]
- **Author**: Koray Aman Arabzadeh
- **Academic Year**: 2024-04-01

### Additional Resources:

- **Demonstration Video**: [Link to Video](https://www.youtube.com/watch?v=-3TfoUi6oTk)

## Running the Project:

1. Clone the project repository.
2. Install Python and required libraries via `pip install -r requirements.txt`.
3. Execute the script with `python app.py` and navigate to `http://localhost:5000` to interact with the web application.
4. Submit email texts for classification and explore the application's ability to detect phishing attempts through the power of NLP and ML.

This project underscores the intersection of cybersecurity, NLP, and ML, offering a practical solution to a prevalent cyber threat while serving as a valuable learning tool for students and professionals alike.
