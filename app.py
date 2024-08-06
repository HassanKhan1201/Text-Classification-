from flask import Flask, request, render_template
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

# Load the dataset and train the model (this can be optimized)
def load_data_from_directory(directory):
    data = []
    target = []
    target_names = []
    
    for idx, category in enumerate(os.listdir(directory)):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            target_names.append(category)
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        data.append(content)
                        target.append(idx)
    
    return data, target, target_names

data_dir = 'C:/Users/muham/Downloads/20_newsgroups'  # Replace with your directory path
X, y, target_names = load_data_from_directory(data_dir)

# Ensure target_names is correct
print(f"Loaded categories: {target_names}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to bag-of-words
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_bow, y_train)

@app.route('/')
def home():
    return render_template('index.html', categories=target_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message:
            return render_template('index.html', prediction="Please enter some text to classify.", categories=target_names)
        try:
            data = [message]
            vect = vectorizer.transform(data).toarray()
            prediction = nb_classifier.predict(vect)
            print(f"Prediction index: {prediction[0]}, Category: {target_names[prediction[0]]}")
            return render_template('index.html', prediction=target_names[prediction[0]], categories=target_names)
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('index.html', prediction=f"Error: {str(e)}", categories=target_names)

if __name__ == '__main__':
    app.run(debug=True)
