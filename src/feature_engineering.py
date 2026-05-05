from pyspark.sql.functions import hour, dayofweek, col, unix_timestamp

def engineer_features(df):
    """
    Perform feature engineering on the cleaned taxi data.
    """
    # 1. Extract time-based features
    df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
    df = df.withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
    
    # 2. Calculate trip duration in minutes
    df = df.withColumn("trip_duration", 
                       (unix_timestamp(col("tpep_dropoff_datetime")) - 
                        unix_timestamp(col("tpep_pickup_datetime"))) / 60)
    
    # Filter out unrealistic durations (e.g., > 3 hours or <= 0)
    df = df.filter((col("trip_duration") > 0) & (col("trip_duration") < 180))
    
    # 3. Select relevant columns for modeling
    model_cols = [
        "VendorID", "RatecodeID", "passenger_count", "trip_distance", 
        "payment_type", "pickup_hour", "day_of_week", "trip_duration", "fare_amount"
    ]
    
    return df.select(model_cols)

if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data
    
    df = load_raw_data()
    df_cleaned = clean_data(df)
    df_features = engineer_features(df_cleaned)
    df_features.show(5)
    print(f"Features Count: {df_features.count()}")
