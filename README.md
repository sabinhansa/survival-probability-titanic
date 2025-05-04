# 🎯 Titanic Survival Probability Prediction

This project is a machine learning solution to the classic **Titanic - Machine Learning from Disaster** problem, where the goal is to predict the **survival probability of passengers** aboard the RMS Titanic based on their attributes.

The model is trained using data from the [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/), and employs preprocessing, feature engineering, and ensemble learning (Random Forest) to achieve high prediction accuracy.

---

## 📊 Problem Statement

Given features such as a passenger’s age, sex, ticket class, and more, predict the **probability that they survived** the Titanic shipwreck.

This is a **binary classification task** (`Survived = 0 or 1`), with the following objectives:

- Clean and preprocess the data.
- Engineer meaningful features.
- Train a robust ML model.
- Predict survival probabilities on unseen data.

---

## 🧱 Project Structure

- **`train.csv` / `test.csv`**  
  Standard Titanic datasets with labeled training data and unlabeled test data.

- **`model.py`**  
  Core training script:
  - Loads and preprocesses the data.
  - Handles missing values and encodes categorical variables.
  - Trains a **Random Forest Classifier**.
  - Predicts probabilities (`predict_proba`) instead of hard classes.
  - Saves results as a CSV for submission.

- **`submission.csv`**  
  Output file containing the predicted **probability of survival** for each test sample.
...

## 🔍 Key Techniques

- **Missing Value Handling**:  
- `Age` and `Fare`: Imputed using median.
- `Embarked`: Filled with mode.
- `Cabin`: Dropped due to high sparsity.

- **Feature Engineering**:  
- Extracted `Title` from names (e.g., Mr, Miss, Dr).
- Created `FamilySize` = SibSp + Parch.
- Created binary `IsAlone` feature.
- Mapped `Sex`, `Embarked`, and `Title` to numeric codes.

- **Model Used**:  
- `RandomForestClassifier` from `sklearn.ensemble`
- Trained with:
  - 100 trees
  - Default depth/criteria (suited for small datasets)
- Outputs **probability** of survival, allowing for more nuanced evaluation.

---

## ✅ Example Prediction Output

| PassengerId | Survived (Probability) |
|-------------|------------------------|
| 892         | 0.081                  |
| 893         | 0.973                  |
| 894         | 0.115                  |
| ...         | ...                    |

---

## 📌 Notes

- The model intentionally predicts **survival probability**, not binary values, for more flexible use (e.g. risk scoring).
- Evaluation was performed locally using cross-validation on the training data.
- The dataset is small and imbalanced; tree-based methods like Random Forest perform well without heavy tuning.

---

## 📁 Requirements

- Python 3.7+
- pandas
- scikit-learn
- numpy

Install dependencies with:

```bash
pip install -r requirements.txt
