# Machine Failure Prediction Using Deep Learning

## Overview

This project uses a deep learning neural network to predict whether a machine is likely to fail based on machine operational and sensor-related features.

## Problem Statement

Unexpected machine failures can lead to production downtime, maintenance costs, and reduced productivity. This project aims to build a machine learning system that can identify potential machine failures from historical machine data.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Jupyter Notebook

## Project Workflow

1. Data loading
2. Data cleaning and preprocessing
3. Exploratory Data Analysis
4. Feature encoding
5. Train-test split
6. Feature scaling using StandardScaler
7. Neural network construction
8. Model training
9. Training and validation analysis
10. Model evaluation
11. Confusion matrix
12. Classification report

## Deep Learning Model

The model uses a fully connected Artificial Neural Network (ANN) with:

- Dense hidden layers
- ReLU activation
- Sigmoid output layer
- Binary classification

The output predicts:

- `0` → No Machine Failure
- `1` → Machine Failure

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training and Validation Curves

Special attention was given to the minority failure class because correctly identifying failures is important in predictive maintenance.

## Results

The final model achieved high overall test accuracy while also being evaluated using class-specific precision, recall, and F1-score to better understand its ability to identify machine failures.

## Files

- `machine_failure_prediction.ipynb` — Complete project notebook
- `machine_failure_model.keras` — Trained deep learning model
- `machine_failure_scaler.pkl` — Saved feature scaler
- `requirements.txt` — Required Python libraries

## Conclusion

This project demonstrates an end-to-end deep learning workflow for machine failure prediction, including preprocessing, neural network training, validation, evaluation, and handling of class imbalance.
