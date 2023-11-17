# Customer Churn Prediction using Keras and Flask

## Overview
This project focuses on predicting customer churn using machine learning techniques implemented with Keras and served through a Flask web application. The predictive model is developed using a neural network with multiple layers, and the web interface allows users to input customer data and get predictions about potential churn.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Deployment](#model-deployment)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

  
## Introduction
This project aims to predict customer churn, which refers to customers discontinuing their association with a company. The code uses machine learning techniques, specifically a neural network implemented in Keras, and a Flask web application for user interaction. The neural network model is trained to predict potential churn by analyzing customer data.

## Features
- Predicts customer churn likelihood
- Implements a Flask web interface for user interaction
- Utilizes a neural network-based predictive model
- Visualizes feature importance for churn prediction

## Installation
To run this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone [repository_url]
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Load the Google Drive for data access as indicated in the code.

## Usage
To use the project:

1. Navigate to the project directory.
2. Start the Flask application using the provided code snippets.
3. Access the web interface in your browser to input customer data and obtain churn predictions.

## Model Training
The code includes steps for exploratory data analysis, relevant feature selection, correlation analysis, scaling, and model creation using Keras Functional API. It involves using RandomForestClassifier for feature importance, MLPClassifier, and Keras to create and train the neural network model.

## Model Deployment
The model deployment utilizes Flask and the trained Keras model. The provided code snippets demonstrate the training process and saving the trained models.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Keras
- TensorFlow
- Flask
- Google Colab (for data loading)

## File Structure
The project structure includes various Python scripts for data analysis, model training, and deployment. Key files include:
- `app.py`: Flask application script
- `model.py`: Code for model training and evaluation
- `CustomerChurn_dataset.csv`: Dataset used for training and analysis
- Other relevant files for data processing and visualization

## Acknowledgments
The code includes dependencies on various Python libraries, frameworks, and platforms. Additionally, it references Google Colab for data loading and model training
