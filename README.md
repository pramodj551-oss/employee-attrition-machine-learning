# Employee Attrition Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-brightgreen)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)
![License](https://img.shields.io/badge/License-MIT-success)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

**A Production-Ready Machine Learning Pipeline for predicting employee attrition using Python and Scikit-learn.**

---

## Project Overview

Employee attrition is a major concern for organizations because losing experienced employees leads to increased hiring costs, training expenses, and reduced productivity.

This project builds a complete Machine Learning pipeline that predicts whether an employee is likely to leave the organization based on HR-related features.

The repository represents **Part 2** of the End-to-End Applied AI & ML Data Product Capstone Project.

---

## Business Problem

Organizations want to answer the following predictive analytics question:

> Can we predict whether an employee is likely to leave the organization before attrition happens?

This project helps HR teams identify employees at risk and enables proactive retention strategies.

---

## Analytics Perspective

This project focuses on:

- **Predictive Analytics**

**Business Question:**
- What is likely to happen?

---

## Objectives

- Build a production-ready Machine Learning pipeline.
- Clean and preprocess employee data.
- Perform feature engineering.
- Train multiple Machine Learning models.
- Compare model performance.
- Select the best-performing model.
- Save the trained model for future predictions.
- Generate business insights.

---

## Dataset

**Dataset Name:** IBM HR Analytics Employee Attrition & Performance

**Dataset Size:**
- Rows: 1470
- Columns: 35

**Dataset Location:** `data/raw/employee_attrition.csv`

---

## Repository Structure

```
employee-attrition-machine-learning/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   └── Employee_Attrition_ML.ipynb
│
├── outputs/
│   ├── charts/
│   ├── metrics/
│   └── reports/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── prediction.py
│   └── utils.py
│
├── models/
│   └── best_model.pkl
│
├── tests/
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
└── main.py
```

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Plotly
- Jupyter Notebook
- Git
- GitHub

---

## Machine Learning Workflow

```
Raw Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization (.pkl)
      │
      ▼
Prediction Pipeline
```

---

## Machine Learning Models

The following supervised learning algorithms are implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier

---

## Model Evaluation

Each model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## Feature Engineering

The pipeline includes:

- Missing Value Handling
- Duplicate Removal
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Train-Test Split

---

## Outputs

The project automatically generates:

- Trained ML Models
- Prediction Reports
- Performance Metrics
- Feature Importance Charts
- Confusion Matrix
- ROC Curve
- Evaluation Reports

---

## Installation

**Clone the repository**
```bash
git clone https://github.com/pramodj551-oss/employee-attrition-machine-learning.git
```

**Move into project directory**
```bash
cd employee-attrition-machine-learning
```

**Create virtual environment**
```bash
python -m venv venv
```

**Activate virtual environment (Windows)**
```bash
venv\Scripts\activate
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## Business Value

The trained model helps organizations:

- Predict employee attrition.
- Reduce employee turnover.
- Improve retention strategies.
- Support HR decision-making.
- Reduce recruitment costs.

---

## Future Improvements

- Hyperparameter Tuning
- Cross Validation
- XGBoost Integration
- LightGBM Integration
- MLflow Experiment Tracking
- Docker Deployment
- Cloud Deployment
- REST API Integration

---

## Project Status

**Part 2 Completed**

**Next Repository:** Part 3 – Interactive Analytics Dashboard

---

## Author

**Pramod Prakash Jadhav**
AI/ML Developer | Security Analyst

📧 Email: pramodj551@gmail.com
💻 GitHub: https://github.com/pramodj551-oss
💼 LinkedIn: https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281

**Repository:** https://github.com/pramodj551-oss/employee-attrition-machine-learning

---

## Acknowledgements

- IBM HR Analytics Dataset
- Scikit-learn Documentation
- Pandas Documentation
- NumPy Documentation
- Matplotlib
- Plotly

---

## License

This project is licensed under the MIT License.

---

## Submission Checklist

- [x] Production Folder Structure
- [x] Machine Learning Pipeline
- [x] Feature Engineering
- [x] Multiple ML Models
- [x] Model Evaluation
- [x] Saved Model
- [x] Prediction Pipeline
- [x] README
- [x] requirements.txt
- [x] MIT License
- [x] GitHub Ready
