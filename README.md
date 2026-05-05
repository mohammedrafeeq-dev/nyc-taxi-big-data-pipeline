# NYC Taxi Trip Fare Prediction (Big Data Pipeline)

🚀 **A Production-Grade Big Data Machine Learning Pipeline**

This project demonstrates a complete end-to-end Big Data pipeline using **PySpark** and **Streamlit** to predict taxi trip fares in New York City. It processes over **2.9 million records** from the official NYC TLC dataset.

## 🎯 Project Overview
The goal is to build a scalable regression model that can accurately estimate the fare amount for a taxi trip based on features like distance, pickup time, location, and passenger count.

## 🏗️ Architecture
- **Data Engine**: PySpark (for scalable data processing and ML)
- **UI Framework**: Streamlit (for real-time model interaction)
- **ML Engine**: PySpark MLlib (Linear Regression, Random Forest, GBT)
- **Deployment**: Modular Python architecture with a production-style structure.

## 📂 Project Structure
```text
big_data_project/
├── app/                # Streamlit application
├── data/               # Raw and processed datasets
├── models/             # Saved ML models
├── notebooks/          # EDA and experimentation
├── plots/              # Visualizations from EDA
├── src/                # Modular source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── utils.py
├── config.py           # Configuration settings
└── requirements.txt    # Project dependencies
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- Java 8 or 11 (required for PySpark)
- `winutils.exe` (if running on Windows - optional but recommended)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Training the Model
To process the data and train the ML models:
```bash
$env:PYTHONPATH = "."
python src/model_training.py
```

### 4. Running the Dashboard
To launch the Streamlit prediction app:
```bash
streamlit run app/streamlit_app.py
```

## 📊 Results
The pipeline compares three models:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosted Trees (GBT) Regressor**

The best model is automatically saved and used for real-time predictions in the Streamlit app.

## 📝 License
This project is for educational and portfolio purposes. Data provided by NYC TLC.
