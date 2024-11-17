# GST Analytics Hackathon: Developing a Predictive Model in GST

## Overview
This repository contains the implementation of a predictive model for the **GSTN Hackathon** analytics challenge. The goal was to construct a model `Fθ(X) → Ypred` that accurately estimates the target variable `Yi` for unseen inputs `Xi`. This project focused on advanced data preprocessing, handling imbalanced datasets, and developing an optimal neural network solution.

---

## Problem Statement

**Dataset Provided:**
- **Training Data (Dtrain):** Matrix of dimension `R(m × n)`
- **Test Data (test):** Matrix of dimension `R(m1 × n)`
- **Target Variables:**
  - `Ytrain`: Dimension `R(m × 1)`
  - `Ytest`: Dimension `R(m1 × 1)`

**Objective:** Develop a model that accurately predicts `Ypred` for unseen test data.

---

## Data Preprocessing

1. **Dropping Unnecessary Columns:**
   - Removed **ID** (irrelevant for predictions).
   - Dropped **Column9** due to excessive missing values.

2. **Selective Row Removal:**
   - Dropped rows where the target variable was `0` and contained null values in key columns.

3. **Handling Missing Values:**
   - Applied **K-Nearest Neighbors (KNN) Imputer** to fill missing values based on similar data points.

4. **Balancing the Dataset:**
   - Used random sampling to create a balanced training set, addressing class imbalance for better model performance.

---

## Model Training

### 1. Neural Network with **MLPClassifier**
- **Architecture:**
  - Hyperparameters tuned using Grid Search.
  - Explored configurations of hidden layers, activation functions, and solvers.
- **Results on Test Data:**
  - Accuracy: **97.58%**
  - Precision: **0.83**
  - Recall: **0.93**
  - F1-Score: **0.88**
  - AUC-ROC: **0.99**

### 2. Cross-Validation with Stratified K-Fold
- Ensured balanced class distribution in each fold.
- **Mean Results on Training Data:**
  - Accuracy: **98.20%**
  - Precision: **0.92**
  - Recall: **0.95**
  - F1-Score: **0.93**

- **Test Data Results:**
  - Accuracy: **97.58%**
  - Precision: **0.83**
  - Recall: **0.93**
  - F1-Score: **0.88**
  - AUC-ROC: **0.99**

### 3. Grid Search Optimization
- Fine-tuned hyperparameters including hidden layer sizes, activation functions, solvers, and learning rates.
- **Improved Results on Test Data:**
  - Accuracy: **97.60%**
  - Precision: **0.84**
  - Recall: **0.93**
  - F1-Score: **0.89**
  - AUC-ROC: **0.99**

---

## Key Evaluation Metrics

1. **Accuracy (97.58%)**
   - The model correctly classified 97.58% of test instances.

2. **Precision (0.84)**
   - 84% of instances predicted as positive were true positives, reducing false positives.

3. **Recall (0.93)**
   - 93% of actual positive cases were correctly detected, minimizing false negatives.

4. **F1-Score (0.89)**
   - Balances precision and recall, reflecting overall performance.

5. **AUC-ROC (0.99)**
   - Indicates excellent model capability in distinguishing between positive and negative classes.

---

## Conclusion
- The **Neural Network model**, enhanced with **Stratified K-Fold Cross-Validation** and **Grid Search**, outperformed other approaches (e.g., linear and logistic regression).
- Achieved high performance metrics, making it robust for deployment in GST-related predictive analytics.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/raged-pineapple/GST-Analytics-Hackathon.git
