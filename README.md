# Diabetes Prediction Using Machine Learning

This repository contains a **clean, modular, and reproducible** machine learning pipeline for predicting diabetes using clinical features.  
The project is designed following **ML best practices**, with proper cross-validation, pipeline-based preprocessing, and model tuning.

---

## 📌 Project Overview

The goal of this project is to predict whether a patient has diabetes based on medical measurements such as glucose level, BMI, age, etc.

Key characteristics of this project:

- No data leakage (scaling is done inside pipelines)
- Proper cross-validation
- Multiple models compared fairly
- Best model selected using **F1-score**
- Hyperparameter tuning applied only to the winning model
- Final model saved for real-world usage

---

## 📊 Dataset

- **File:** `data/diabetes.csv`
- **Source:** Pima Indians Diabetes Dataset
- **Target column:** `Outcome`  
  - `0` → No diabetes  
  - `1` → Diabetes

### Features

| Feature | Description |
|------|-----------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-Hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

---

## 🗂 Project Structure

```
ML_Diabets_1404/
│
├── data/
│   └── diabetes.csv
│
├── src/
│   ├── data_loader.py      # Load dataset
│   ├── models.py           # Model pipelines
│   ├── evaluate.py         # Cross-validation metrics
│   ├── tune.py             # Hyperparameter tuning
│   ├── train.py            # Training & model selection
│   └── predict.py          # Inference script
│
├── models/
│   └── best_model.joblib   # Final trained model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Train and select the best model

```bash
python src/train.py
```

This will:

- Compare multiple models using cross-validation
- Select the best model based on **F1-score**
- Apply hyperparameter tuning to the winning model
- Save the final model to `models/best_model.joblib`

---

### 2️⃣ Make a prediction

```bash
python src/predict.py
```

The prediction script:
- Loads the trained model
- Accepts a sample input as a **pandas DataFrame**
- Outputs prediction and class probabilities

---

## 🧠 Models Used

The following models are evaluated:

- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)

⚠️ Only the **best-performing model** is tuned and kept.

---

## 📈 Evaluation Strategy

Models are evaluated using **5-Fold Cross-Validation** with the following metrics:

- Accuracy
- F1-score (primary decision metric)
- ROC-AUC

Why F1-score?
- The dataset is mildly imbalanced
- F1 balances precision and recall
- Accuracy alone can be misleading

---

## 🔍 Hyperparameter Tuning

- Implemented using `GridSearchCV`
- Applied **only to the best model**
- Prevents overfitting and unnecessary computation
- Uses F1-score as the optimization metric

---

## 🛡️ Design Decisions

- **Pipelines** are used to avoid data leakage
- **StandardScaler** is applied only where appropriate
- Feature names are preserved during inference
- No notebook dependency for execution
- Code is modular and reusable

---

## 📦 Model Persistence

The final trained model is saved using `joblib`:

```
models/best_model.joblib
```

This model can be directly used in:
- APIs (FastAPI / Flask)
- Web apps (Streamlit)
- Production inference pipelines

---

## 🔮 Future Improvements

- Handle class imbalance explicitly (SMOTE / class weights)
- Threshold tuning based on ROC curve
- REST API deployment (FastAPI)
- Model explainability (SHAP)

---

## 📜 License

This project is licensed under the **GPL-3.0 License**.

---

## 👤 Author

This project was developed as a **clean and defensible machine learning implementation**, focusing on correctness rather than superficial results.

