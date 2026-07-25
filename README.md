## 💰 Smart Finance Tracker & Predictor

  An enterprise-grade personal finance and predictive cash outflow application built with Python, Streamlit, SQLite, and Scikit-Learn.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📌 Executive Summary

  Modern financial trackers treat transaction history as static records. This system turns historical financial data into an active machine learning ecosystem. The application integrates real-time exchange rate normalization, an embedded relational SQL database engine, and a 4-way parallel machine learning training matrix to forecast future 7-day cash outflows.

---

## 🛠️ Key System Architecture

[ Ingestion & API Layer ] ──> [ SQLite Relational Engine ] ──> [ Feature Engineering ] ──> [ 4-Way ML Matrix ] ──> [ 7-Day Forecast ]

### 1. Ingestion & Real-Time Normalization
* Multi-currency input interface (USD, THB).
* Live currency conversion integrated via REST API (`open.er-api.com`) to maintain unit consistency across the database.

### 2. Relational Database Management (SQLite)
* Structured relational schema utilizing `sqlite3`.
* Primary key `ID` indexing for explicit SQL record transactions and deletion operations.

### 3. Feature Engineering & Signal Processing
* Converts unstructured date stamps into temporal machine learning signals:
  * `DayOfMonth`
  * `DayOfWeek`
  * `IsWeekend`
  * `Rolling_7D_Avg` (Moving average momentum indicator)

### 4. Multi-Model ML Benchmarking
Evaluates four regression algorithms in parallel to identify spending behavior patterns:
* **Multiple Linear Regression (MLR)** - Baseline linear scaling model.
* **Support Vector Regression (SVR)** - Radial Basis Function (RBF) kernel mapping.
* **Random Forest Regressor** - Decision tree ensemble averaging.
* **Gradient Boosting Regressor** - Sequential residual boosting ensemble.

---

## 📊 Performance Metric Evaluation

Models are dynamically fitted on historical ledger transactions and evaluated using the **Coefficient of Determination ($R^2$ Score)**:

| Model | Model Class | Characteristics |
| :--- | :--- | :--- |
| **Linear Regression** | Parametric | Fast baseline model for linear trends |
| **Support Vector Regression** | Non-Parametric Kernel | Robust against daily volatility |
| **Random Forest** | Bagging Ensemble | Effective at conditional rules (e.g., weekend spikes) |
| **Gradient Boosting** | Boosting Ensemble | Learns complex, non-linear human spending habits |

---

## 💻 Local Quickstart Guide

### Prerequisites
* Python 3.9+ installed on your machine.

### Installation & Run

1. Clone this repository:
   ```bash
   git clone [https://github.com/Mikk89/smart-finance-tracker.git](https://github.com/Mikk89/smart-finance-tracker.git)
   cd smart-finance-tracker


1.  Install dependencies:

    pip install streamlit pandas numpy requests plotly scikit-learn

2.  Launch the Streamlit application:

    streamlit run app.py

## 👨‍💻 Author

Developed as a Computer Engineering portfolio project demonstrating full-stack data engineering, API integration, and applied machine learning pipelines.




