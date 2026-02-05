# 🎗️ Breast Cancer Detection using Machine Learning  
### Accurate Classification of Malignant vs Benign Tumors

<div align="center">
  <img src="banner.png" alt="Breast Cancer Detection using Machine Learning" width="95%">
</div>

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Classification](https://img.shields.io/badge/Task-Classification-purple?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Project Overview

The **Breast Cancer Detection** project is a machine learning classification system designed to **accurately distinguish between malignant and benign tumors** using diagnostic features extracted from breast mass images.

This project demonstrates a **complete end-to-end ML workflow**, including:
- exploratory data analysis (EDA),
- feature engineering,
- preprocessing with pipelines,
- cross-validation,
- multi-model comparison,
- and final model selection.

It serves as both an **educational reference** and a **practical healthcare ML example**.

---

## Why This Project Matters

Early and accurate detection of breast cancer plays a critical role in improving survival rates and treatment outcomes.  
Machine learning provides **fast, consistent, and data-driven decision support**, reducing reliance on manual interpretation alone.

This project shows how classical ML algorithms can:
- achieve **very high diagnostic accuracy**,  
- minimize false negatives,  
- and remain interpretable and reproducible.

---

## Dataset Overview

This project uses the **built-in Breast Cancer dataset from scikit-learn**, containing real diagnostic measurements computed from digitized images of breast mass biopsies.

### Target Classes
- **0** → Malignant  
- **1** → Benign  

### Dataset Characteristics
- 569 total samples  
- 30 numerical diagnostic features  
- Clean dataset with no missing values  
- Widely used benchmark dataset in medical ML research  

---

## Repository Structure

```text
.
📁 Breast_Cancer_Detection/
├── banner.png
├── README.md
├── LICENSE
├── requirements.txt
├── notebook/
│   └── breast_cancer_detection.ipynb
└── models/
    └── best_breast_cancer_model.pkl
````

---

## Feature Engineering & Preprocessing

### 1. Exploratory Analysis

* Class distribution visualization
* Feature scatter plots to inspect separability
* Correlation-based feature insight

### 2. Feature Creation

Additional features were created to demonstrate encoding techniques:

* **Ordinal feature**: Tumor size category (`Small`, `Medium`, `Large`)
* **Nominal feature**: Texture type (`Smooth`, `Rough`)

### 3. Preprocessing Pipeline

* **StandardScaler** for numerical features
* **OrdinalEncoder** for ordered categorical features
* **OneHotEncoder** for nominal categorical features

All transformations are combined using a **ColumnTransformer**, ensuring clean and reproducible preprocessing.

---

## Model Training & Evaluation

Multiple classification algorithms were trained and evaluated using **5-fold cross-validation** and a held-out test set.

### Models Evaluated and Performance

| Model                  | CV Accuracy | Test Accuracy |
| ---------------------- | ----------- | ------------- |
| Logistic Regression    | 97.80%      | 97.37%        |
| Ridge (L2) Regression  | 97.80%      | 97.37%        |
| Lasso (L1) Regression  | 97.36%      | **98.25%**    |
| Support Vector Machine | 97.14%      | 97.37%        |
| K-Nearest Neighbors    | 96.48%      | 97.37%        |
| Decision Tree          | 92.09%      | 91.23%        |
| Random Forest          | 95.82%      | 94.74%        |

### Final Model Selection

The **Lasso (L1) Logistic Regression** model achieved the **highest test accuracy (98.25%)** and was selected as the final model.

The trained model is saved as:

```text
best_breast_cancer_model.pkl
```

---

## Results Summary

* Excellent separation between malignant and benign classes
* Very high classification accuracy (>98%)
* Strong generalization through cross-validation
* Clean, production-ready ML pipeline

This confirms that even **simple, well-regularized models** can perform exceptionally well on medical classification tasks.

---

## Run Locally

```bash
git clone https://github.com/harisyar-ai/breast-cancer-detection.git
cd breast-cancer-detection
pip install -r requirements.txt
```

Open the notebook and run all cells to reproduce results.

---

## Future Improvements

* Add ROC curves and AUC scores
* Integrate SHAP for model explainability
* Deploy the model as a web application
* Explore ensemble or hybrid models

---

```
Developed by Muhammad Haris Afridi
February 2026

Stars ⭐ and feedback are highly appreciated
github.com/harisyar-ai
```
