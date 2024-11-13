# PIMA Diabetes Prediction using Keras

This project aims to predict the presence of diabetes in patients using the PIMA Indians Diabetes dataset. The dataset includes various medical and demographic attributes of the patients. The goal is to build and evaluate a neural network model using Keras, applying techniques like data normalization, early stopping, and learning rate adjustment to optimize performance.

## Project Overview

1. **Dataset**: 
   - The PIMA Indians Diabetes dataset is sourced from the UCI Machine Learning Repository. It contains 768 samples with 8 features related to medical history and physical measurements.
   - **Features**:
     - `times_pregnant`: Number of times the patient has been pregnant.
     - `glucose_tolerance_test`: Plasma glucose concentration after a 2-hour oral glucose tolerance test.
     - `blood_pressure`: Diastolic blood pressure (mm Hg).
     - `skin_thickness`: Triceps skinfold thickness (mm).
     - `insulin`: 2-hour serum insulin (mu U/ml).
     - `bmi`: Body mass index (weight in kg/(height in m)^2).
     - `pedigree_function`: Diabetes pedigree function (a score indicating genetic influence on diabetes).
     - `age`: Age of the patient (years).
   - **Target Variable**: `has_diabetes` - A binary outcome where `1` indicates the presence of diabetes and `0` indicates the absence.

2. **Objective**: 
   - Build a neural network using Keras to classify patients as diabetic or non-diabetic.
   - Optimize the model’s performance using techniques like normalization, early stopping, and adaptive learning rates.

## Steps in the Project

### 1. Data Preparation
   - **Data Loading**: Load the dataset and inspect its structure.
   - **Splitting**: Split the data into training and testing sets.
   - **Normalization**: Apply standard scaling to ensure the features have similar ranges.

### 2. Model Building
   - **Architecture**: The model is a simple feedforward neural network with two hidden layers, each with 12 neurons and ReLU activation.
   - **Regularization**: L2 regularization is added to prevent overfitting.
   - **Output Layer**: A sigmoid activation function is used to produce a probability for binary classification.

### 3. Model Training
   - **Optimizer**: Stochastic Gradient Descent (SGD) with an initial learning rate.
   - **Callbacks**: 
     - **Early Stopping**: Monitors validation loss and stops training if there is no improvement after 10 epochs.
     - **ReduceLROnPlateau**: Reduces the learning rate if the validation loss plateaus, preventing the model from getting stuck in a local minimum.

### 4. Model Evaluation
   - **Metrics**: Evaluate the model using accuracy and the ROC-AUC score to measure the trade-off between sensitivity and specificity.
   - **Plots**: 
     - Plot the training and validation loss over epochs.
     - Plot the training and validation accuracy.
     - Plot the ROC curve to visualize model performance.

## Results
- **Accuracy**: The final accuracy on the test set was approximately 0.797.
- **ROC-AUC**: The ROC-AUC score achieved was 0.821, indicating the model's ability to distinguish between positive and negative cases.

## Conclusion
The neural network model achieved good performance metrics with the given dataset. However, given the relatively small size of the dataset (768 samples), further testing on larger and more varied datasets is recommended for robustness.
