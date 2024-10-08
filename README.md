# **Feature Engineering and Diabetes Prediction**

This project focuses on predicting diabetes outcomes by applying **feature engineering** techniques to the **diabetes dataset**.

## **Files**

- **`Feature Engineering.py`**: A Python script that performs data preprocessing, feature engineering, and model training for diabetes prediction.
- **`diabetes.csv`**: The dataset containing patient health information related to diabetes.

## **Project Overview**

The goal of this project is to predict whether a patient has diabetes using various feature engineering techniques and a machine learning model. The main steps include:

### 1. **Data Loading**
- Load the **`diabetes.csv`** dataset.

### 2. **Feature Engineering**
- Identifying and handling **outliers**.
- Handling missing values using **KNN Imputation**.
- Creating new features such as **Weight** and **Age_Class** based on BMI and age values.
- **One-hot encoding** for categorical variables.
- **Scaling** numerical features.

### 3. **Model Training**
- The target variable (**Outcome**) is predicted using a **RandomForestClassifier**.
- The model is trained and tested on the dataset, and its accuracy is evaluated.

