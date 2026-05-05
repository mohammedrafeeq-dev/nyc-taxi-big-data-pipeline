import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_spark_session
from src.data_loader import load_raw_data

def run_eda():
    """
    Perform EDA on the loaded Spark DataFrame.
    """
    df = load_raw_data()
    
    # 1. Basic Stats
    print("--- Basic Statistics ---")
    df.select("trip_distance", "fare_amount", "tip_amount", "total_amount").describe().show()
    
    # 2. Missing Values
    print("--- Missing Values ---")
    from pyspark.sql.functions import col, count, when
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    # 3. Distribution Analysis (sample for plotting)
    print("--- Sampling for Visualizations ---")
    sample_df = df.select("trip_distance", "fare_amount", "passenger_count").sample(False, 0.01).toPandas()
    
    # Create plots
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_df['fare_amount'], bins=100, kde=True)
    plt.title('Distribution of Fare Amount')
    plt.xlim(0, 100)
    plt.savefig('plots/fare_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='trip_distance', y='fare_amount', data=sample_df)
    plt.title('Trip Distance vs Fare Amount')
    plt.xlim(0, 50)
    plt.ylim(0, 200)
    plt.savefig('plots/distance_vs_fare.png')
    plt.close()
    
    print("EDA Visualizations saved to 'plots/' directory.")

if __name__ == "__main__":
    run_eda()
