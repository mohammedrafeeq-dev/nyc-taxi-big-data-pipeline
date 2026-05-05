import os
from src.utils import get_spark_session
from config import DATA_RAW_DIR, RAW_DATA_FILE

def load_raw_data():
    """
    Load the raw NYC Taxi dataset into a Spark DataFrame.
    """
    spark = get_spark_session()
    file_path = os.path.join(DATA_RAW_DIR, RAW_DATA_FILE)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    df = spark.read.parquet(file_path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Total Records: {df.count()}")
    df.printSchema()
    df.show(5)
