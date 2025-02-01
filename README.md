# Dataset Analysis - Supervised Learning

## Overview
This project involves the analysis of a dataset using various supervised learning techniques. The goal is to explore different machine learning models, tune their hyperparameters, and compare their performance. The models used include Softmax Regression, Decision Trees, Random Forests, AdaBoost, Soft Voting, and a Blender model.

## Project Structure
1. **Dataset Exploration and Preprocessing**
2. **Model Training and Evaluation**
   - Softmax Regression
   - Decision Tree
   - Random Forest
   - AdaBoost
   - Soft Voting
   - Blender
3. **Comparison of Models**

## Dataset Exploration and Preprocessing
- **Dataset Loading**: The dataset was loaded and explored for missing values and duplicates. No missing values or duplicates were found.
- **Data Encoding**: Categorical columns with Yes/No values were encoded using `LabelBinarizer`, and columns with more values were encoded using `LabelEncoder`. The `gormiti_land` column was not encoded initially but will be encoded for ensemble methods.
- **Outlier Handling**: Outliers were considered for handling if they impacted the models later.

## Model Training and Evaluation

### 1. Softmax Regression
- **Preprocessing**: No further preprocessing was necessary.
- **Training**: A logistic regression model was trained using `GridSearchCV` with cross-validation. The parameter grid for `C` was `[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]`. The best `C` value was found to be 100.
- **Evaluation**: The model achieved an accuracy of 0.75 on both the training and test sets, indicating no overfitting.

### 2. Decision Tree
- **Hyperparameter Tuning**: The following parameters were tuned using `GridSearchCV`:
  - `max_depth`: [2, 5, 7, 10, 20, 25, 30, 40, 50, 100, 125, 150, 200]
  - `criterion`: ['gini', 'entropy']
  - `min_samples_split`: [1, 3, 5]
  - `min_samples_leaf`: [1, 3, 5]
  - `max_features`: list(range(0,13))
  - `max_leaf_nodes`: [10, 20, 30]
- **Best Parameters**:
  - `{'criterion': 'gini', 'max_depth': 10, 'max_features': 11, 'max_leaf_nodes': 30, 'min_samples_leaf': 2, 'min_samples_split': 1}`
- **Evaluation**: The model achieved an accuracy of 0.789 on the training set and 0.7715 on the test set, indicating slight overfitting.

### 3. Random Forest
- **Hyperparameter Tuning**: The best parameters were:
  - `{'max_depth': 11, 'max_leaf_nodes': 70, 'n_estimators': 200}`
- **Feature Importance**: The most important features were `Against_Lava` and `Against_Rock/Against_wind`.
- **Evaluation**: The model achieved an accuracy of 0.8315 on the test set and 0.246125 on the training set, indicating slight overfitting.

### 4. AdaBoost
- **Hyperparameter Tuning**: The following parameters were tuned:
  - `n_estimators`: [10, 30, 50, 100, 200, 400]
  - `learning_rate`: [0.001, 0.01, 0.1, 1, 5, 10]
- **Evaluation**: The model achieved similar accuracy on both the training and test sets, with a difference of less than 0.005, indicating no overfitting.

### 5. Soft Voting
- **Models Included**:
  - Logistic Regression
  - Softmax Regression
  - Random Forest
  - Decision Tree
- **Evaluation**: The voting classifier achieved an accuracy of 0.8195, performing slightly better than most individual models except Random Forest.

### 6. Blender
- **Model Composition**: The blender model was composed of the best estimators from Softmax Regression, Decision Tree, Random Forest, and AdaBoost.
- **Evaluation**: The model achieved an accuracy of 0.85 on the training set and 0.83 on the test set, indicating slight overfitting.

## Comparison of Models
- **Worst Model**: AdaBoost with an accuracy of 0.7415.
- **Best Model**: Random Forest with an accuracy of 0.8315.
- **Second Best Model**: Ensemble with an accuracy of 0.83.

## Conclusion
All models performed comparably well, with Random Forest being the best performer. The ensemble methods (Soft Voting and Blender) also showed competitive performance. However, achieving 100% accuracy may require more data or further model tuning.

## Requirements
- Python 3.x
- Libraries: `sklearn`, `numpy`, `pandas`

## Usage
1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebook or Python script to reproduce the analysis.

## Author
Oleg Lastocichin
