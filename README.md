# 📊 Cross Validation with Different Machine Learning Algorithms

This project demonstrates how to preprocess data and apply **cross-validation** using different machine learning models on a real-world dataset.

---

## 📁 Dataset

**Customer Churn Prediction Dataset**  
Contains information about telecom customers and whether they have churned (left the service).

---

## 🔧 Preprocessing Steps

- ✅ Uploaded dataset using Google Colab
- ✅ Inspected data with `.head()`, `.info()`, `.columns()`
- ✅ Dropped non-useful column: `customerID`
- ✅ Converted `TotalCharges` from object to numeric
- ✅ Identified:
  - Binary categorical columns (e.g., `gender`, `Partner`) → Label Encoded
  - Multi-class columns (e.g., `InternetService`, `Contract`) → One-Hot Encoded
- ✅ Handled missing values in `TotalCharges` using median
- ✅ Prepared features `X` and target `y`

---

## 🧠 Machine Learning Models Used

- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Random Forest Classifier**
- (XGBoost and others can be added later)

---

## 🔁 Cross-Validation

Implemented K-Fold and Stratified K-Fold cross-validation using:

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
