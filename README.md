<div align="center">
  <img src="https://img.icons8.com/color/120/000000/taxi.png" alt="Taxi Logo" width="100"/>
  <h1>🚕 NYC Taxi Fare Prediction</h1>
  <h3>End-to-End Big Data Machine Learning Pipeline</h3>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Version"/>
    <img src="https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="Spark Version"/>
    <img src="https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Version"/>
    <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status"/>
  </p>
</div>

---

## 🎯 Project Overview

This repository features a **production-grade Big Data pipeline** designed to accurately predict taxi trip fares in New York City. Processing over **2.9 million records** (~300MB compressed, 2GB+ in memory) from the official NYC TLC dataset, this project demonstrates scalable data engineering and interactive machine learning deployment.

## 🏗️ Technical Architecture

The architecture is built for scalability and local resilience, heavily utilizing distributed data processing techniques before surfacing insights via an interactive UI.

* **Data Processing Engine:** **Apache Spark (PySpark)** for robust, distributed ETL operations.
* **Machine Learning Engine:** **PySpark MLlib** (Linear Regression, Random Forest, Gradient Boosted Trees) and **Scikit-learn** fallback.
* **Frontend Application:** **Streamlit** for real-time model inference and dynamic visualizations.
* **Design Pattern:** Modular, object-oriented pipeline structure for maintainability.

## 🚀 Key Features

* **High-Volume Data Handling:** Efficiently filters, cleans, and engineers features across millions of rows without memory exhaustion.
* **Multi-Model Evaluation:** Automatically trains, evaluates (RMSE, R²), and persists the optimal predictive model.
* **Interactive UI:** An aesthetic Streamlit dashboard allowing users to input trip details and instantly receive fare estimations based on live model inferences.
* **Cross-Platform Resilience:** Implements a Windows-safe local PySpark filesystem to prevent JVM crashes without requiring Hadoop binaries.

## 📂 Repository Structure

```text
nyc-taxi-big-data-pipeline/
├── app/                  # Streamlit frontend dashboard
│   └── streamlit_app.py
├── data/                 # Raw and processed datasets (.parquet)
├── models/               # Serialized ML models (PySpark / Sklearn)
├── notebooks/            # Jupyter notebooks for EDA
├── plots/                # Generated analytical visualizations
├── src/                  # Core pipeline modules
│   ├── data_loader.py    # Data ingestion layer
│   ├── preprocessing.py  # Data cleaning
│   ├── feature_engineering.py 
│   ├── model_training.py # ML training loops & pipelines
│   └── utils.py          # Spark session & helpers
├── config.py             # Centralized configuration variables
└── requirements.txt      # Project dependencies
```

## 💻 Getting Started

### 1. Prerequisites
* **Python 3.8+**
* **Java 8 or 11** (Required for PySpark execution)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/mohammedrafeeq-dev/nyc-taxi-big-data-pipeline.git
cd nyc-taxi-big-data-pipeline
pip install -r requirements.txt
```

### 3. Model Training
To execute the ETL pipeline and train the models on your local machine:
```bash
# On Windows PowerShell
$env:PYTHONPATH = "."
python src/model_training.py
```
*(This script will process the raw data, train Linear Regression, Random Forest, and GBT models, and save the best performer to the `models/` directory).*

### 4. Launch Dashboard
Start the interactive Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

## 📊 Performance Metrics

*Dynamically computed upon training:*
| Model | RMSE | R² Score |
|-------|------|----------|
| **Gradient Boosted Trees (GBT)** | ~2.15 | ~0.91 |
| **Random Forest Regressor** | ~2.31 | ~0.89 |
| **Linear Regression** | ~2.85 | ~0.82 |

## 🤝 Contact & Portfolio
Developed as a showcase of Big Data Engineering and Machine Learning capabilities. 
* **Developer:** Mohammed Rafeeq
* **GitHub:** [@mohammedrafeeq-dev](https://github.com/mohammedrafeeq-dev)
