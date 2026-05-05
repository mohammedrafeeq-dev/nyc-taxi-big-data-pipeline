from pyspark.sql.functions import col
from src.data_loader import load_raw_data

def clean_data(df):
    """
    Clean the NYC Taxi dataset.
    """
    # 1. Remove duplicates
    df = df.dropDuplicates()
    
    # 2. Filter invalid trips
    # We want trips with positive fare and distance
    df = df.filter(
        (col("fare_amount") > 0) & 
        (col("trip_distance") > 0) & 
        (col("passenger_count") > 0) &
        (col("total_amount") > 0)
    )
    
    # 3. Handle missing values
    # For simplicity, we drop rows with any null values in critical columns
    df = df.dropna(subset=["fare_amount", "trip_distance", "PULocationID", "DOLocationID"])
    
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print(f"Initial Count: {df.count()}")
    df_cleaned = clean_data(df)
    print(f"Cleaned Count: {df_cleaned.count()}")
