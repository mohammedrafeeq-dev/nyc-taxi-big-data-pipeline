import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model")

# File names
RAW_DATA_FILE = "yellow_tripdata_2024-01.parquet"
PROCESSED_DATA_FILE = "taxi_processed.parquet"

# Spark Configuration
SPARK_APP_NAME = "NYCTaxiFarePrediction"
SPARK_MASTER = "local[*]"

# Feature Columns
CATEGORICAL_COLS = ["VendorID", "RatecodeID", "payment_type"]
NUMERICAL_COLS = ["trip_distance", "pickup_hour", "day_of_week", "passenger_count"]
TARGET_COL = "fare_amount"
