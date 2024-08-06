### Text Classification with Flask

This project is a text classification application built using Flask and scikit-learn. It uses a Naive Bayes classifier to categorize text into predefined categories. The frontend allows users to input text and receive a classification result.

## Project Structure


## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Virtual environment (`venv`)

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/HassanKhan1201/Text-Classification-.git
    cd text-classification-task
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure the dataset directory (`20_newsgroups`) is correctly structured:**

    The `20_newsgroups` directory should contain subdirectories for each category, with text files inside each subdirectory.

5. **Run the Flask application:**

    ```bash
    python app.py
    ```

6. **Open the application in your browser:**

    Navigate to `http://127.0.0.1:5000/` to access the frontend.

## Usage

1. Enter a piece of text in the textarea.
2. Click the "Classify" button.
3. The application will display the predicted category for the input text.

## Code Explanation

- **app.py:** This file contains the Flask application code. It includes routes for the homepage and the prediction functionality. The text classification model is trained using the Naive Bayes classifier from scikit-learn.

- **templates/index.html:** This file contains the HTML template for the frontend. It includes a form for text input and a section to display the prediction result.

## Debugging

If you encounter any issues, check the console output for debugging information. The application prints loaded categories and prediction details to help identify any problems.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.


