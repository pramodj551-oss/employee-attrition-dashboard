# Employee Attrition Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **A Production-Ready Interactive Employee Attrition Dashboard built using Streamlit, Machine Learning, and Plotly.**

🔗 **Live Demo:** *(add your Streamlit Cloud URL here)*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Objectives](#objectives)
- [Screenshots](#screenshots)
- [Dashboard Features](#dashboard-features)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Dashboard Workflow](#dashboard-workflow)
- [Installation](#installation)
- [Run Dashboard](#run-dashboard)
- [Dashboard Outputs](#dashboard-outputs)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

# Project Overview

Employee attrition is a major concern for organizations because replacing experienced employees is expensive and time-consuming.

This project provides an interactive Streamlit dashboard that allows HR professionals and business stakeholders to:

- Explore employee data
- Visualize key HR metrics
- Predict employee attrition
- Analyze feature importance
- Generate business insights

This repository represents **Part 3** of the **End-to-End Applied AI & ML Data Product Capstone Project**.

---

# Business Problem

Organizations need answers to questions such as:

- Which employees are at high risk of leaving?
- Which department has the highest attrition?
- Does overtime increase attrition?
- Which age groups leave more frequently?
- How does monthly income impact retention?
- Which factors influence employee attrition the most?

---

# Objectives

- Build an interactive HR analytics dashboard
- Visualize employee attrition trends
- Integrate trained Machine Learning model
- Predict employee attrition
- Display feature importance
- Generate business insights
- Build a production-ready Streamlit application

---

# Screenshots

> *Add screenshots or a short demo GIF of the dashboard here — this is usually the first thing recruiters and stakeholders look for in a dashboard project.*

| Home | EDA | Prediction |
|------|-----|------------|
| *add image* | *add image* | *add image* |

---

# Dashboard Features

## Home

- Project Overview
- Dataset Summary
- Business KPIs

---

## Dataset Explorer

- View Dataset
- Search Records
- Filter Employees
- Download Dataset

---

## Exploratory Data Analysis (EDA)

- Attrition Distribution
- Department-wise Analysis
- Gender Analysis
- Job Role Analysis
- Monthly Income Distribution
- Overtime Analysis
- Correlation Heatmap

---

## Machine Learning

> *Algorithm: specify your best-performing model here (e.g., Random Forest / XGBoost / Logistic Regression).*

- Load Trained Model
- Model Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Attrition Prediction

Users can enter:

- Age
- Gender
- Department
- Job Role
- Monthly Income
- Years at Company
- Overtime
- Job Satisfaction
- Work-Life Balance

Dashboard predicts:

- Attrition (Yes/No)
- Prediction Probability

---

## Feature Importance

Interactive visualization of the most influential features used by the machine learning model.

---

## Reports

Download:

- Prediction Results
- Classification Report
- Dashboard Reports

---

# Dataset

## Dataset Name

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

## Dataset Size

- Rows: **1470**
- Columns: **35**

## Dataset Location

```
data/raw/employee_attrition.csv
```

---

# Repository Structure

```
employee-attrition-dashboard/
│
├── assets/
│   ├── logo.png
│   └── style.css
│
├── data/
│   ├── raw/
│   │   └── employee_attrition.csv
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── Employee_Attrition_Dashboard.ipynb
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── logs/
│
├── src/
│   ├── config.py
│   ├── logger.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dashboard.py
│   ├── predictor.py
│   ├── charts.py
│   ├── utils.py
│   └── __init__.py
│
├── tests/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .env.example
```

---

# Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Joblib

---

# Dashboard Workflow

```
Employee Dataset
        │
        ▼
Load Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Machine Learning Model
        │
        ▼
Interactive Dashboard
        │
        ▼
Prediction
        │
        ▼
Business Insights
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/pramodj551-oss/employee-attrition-dashboard.git
```

Move into project

```bash
cd employee-attrition-dashboard
```

Create virtual environment

```bash
python -m venv venv
```

Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run Dashboard

```bash
streamlit run app.py
```

---

# Dashboard Outputs

The dashboard provides:

- Employee Overview
- Interactive Charts
- Attrition Prediction
- Feature Importance
- Downloadable Reports

---

# Future Improvements

- Explainable AI (SHAP)
- Power BI Dashboard
- Docker Deployment
- Cloud Deployment
- Authentication
- Multi-user Support
- Real-time Prediction API

---

# Project Status

**Part 3 - Employee Attrition Dashboard**

- Interactive Dashboard
- Data Visualization
- Machine Learning Prediction
- Business Insights

---

# Author

**Pramod Prakash Jadhav**
AI/ML Developer | Security Analyst

- 📧 Email: [pramodj551@gmail.com](mailto:pramodj551@gmail.com)
- 💼 LinkedIn: [pramod-prakash-jadhav-42ba2281](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)
- 💻 GitHub: [pramodj551-oss](https://github.com/pramodj551-oss)
- 📂 Repository: [employee-attrition-dashboard](https://github.com/pramodj551-oss/employee-attrition-dashboard)

---

# Acknowledgements

- IBM HR Analytics Dataset
- Streamlit Community
- Scikit-learn Documentation
- Plotly Documentation
- Pandas Documentation
- Python Community

---

# License

This project is licensed under the MIT License.

---

# Submission Checklist

- [x] Interactive Dashboard
- [x] Machine Learning Integration
- [x] Prediction Module
- [x] Interactive Charts
- [x] Download Reports
- [x] Production Folder Structure
- [x] Clean Python Code
- [x] README
- [x] Requirements
- [x] MIT License
- [x] GitHub Ready
