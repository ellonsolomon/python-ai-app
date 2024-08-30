# Iris Flower Classifier

This project implements an Iris flower classification application using TensorFlow, Flask, and Dash. The application allows users to classify Iris flowers based on their features.

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Files](#files)
- [Contributing](#contributing)
- [License](#license)

## About

The Iris flower classifier uses machine learning techniques to classify Iris flowers into three species based on their sepal and petal measurements. This application is built with the following technologies:

- **TensorFlow**: For building and training the classification model.
- **Flask**: To serve the application.
- **Dash**: For creating interactive web applications.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ellonsolomon/python-ai-app.git
   cd python-ai-app

2. Build the Docker image:
   ```bash
   docker build -t pyapp .

3. Running the Application
To run the application, execute the following command:
   ```bash
   docker run -p 8050:8050 pyapp

Now, open your browser and visit:

   ```bash
   http://127.0.0.1:8050/
   ```

# Usage
Once the application is running, you can input the sepal and petal measurements to classify the Iris flower. The model will provide predictions based on the provided data.

# Files
- iris_flask.py: Main entry point for the Flask application.
- iris_frontend.py: Contains the Dash frontend logic.
- base_iris.py: Defines the machine learning model.
- iris_data_encoded.csv: Dataset used for training the model.
- Dockerfile: Configuration file for building the Docker image.
- requirements.txt: Lists the required Python packages.

# Contributing
Contributions are welcome! If you have suggestions for improvements or want to report issues, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.
