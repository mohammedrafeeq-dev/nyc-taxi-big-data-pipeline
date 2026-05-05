import sys
import os
import shutil
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from src.utils import get_spark_session
from src.data_loader import load_raw_data
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from config import BEST_MODEL_PATH


def train_models():
    """
    Train and compare multiple models.
    """
    spark = get_spark_session()

    # 1. Pipeline Stages
    categorical_cols = ["VendorID", "RatecodeID", "payment_type"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
        for c in categorical_cols
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_encoded")
        for c in categorical_cols
    ]

    numerical_cols = [
        "passenger_count",
        "trip_distance",
        "pickup_hour",
        "day_of_week",
        "trip_duration"
    ]

    assembler = VectorAssembler(
        inputCols=[f"{c}_encoded" for c in categorical_cols] + numerical_cols,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features"
    )

    # 2. Data Preparation
    print("Loading and preparing data...")
    df = load_raw_data()
    print("✓ Raw data loaded")
    
    # Sample EARLY (10%) for speed + cache
    df = df.sample(False, 0.1, seed=42).cache()
    df.count()  # Trigger cache
    
    df = clean_data(df)
    print("✓ Data cleaned")
    
    df = engineer_features(df)
    print("✓ Features engineered")
    
    # No need for additional sample, already small
    print(f"✓ Final dataset size: {df.count()} rows")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Evaluators (moved outside loop)
    evaluator_rmse = RegressionEvaluator(
        labelCol="fare_amount",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="fare_amount",
        predictionCol="prediction",
        metricName="r2"
    )

    # 3. Models to compare
    models = {
        "LinearRegression": {
            "model": LinearRegression(featuresCol="scaled_features", labelCol="fare_amount"),
            "use_scaling": True
        },
        "RandomForest": {
            "model": RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=10),
            "use_scaling": False
        }
    }

    best_rmse = float("inf")
    best_model_pipeline = None
    best_model_name = None

    for name, config in models.items():
        print(f"Training {name}...")

        stages = indexers + encoders + [assembler]

        # Add scaler only if needed
        if config["use_scaling"]:
            stages.append(scaler)

        stages.append(config["model"])

        pipeline = Pipeline(stages=stages)

        model_fit = pipeline.fit(train_df)
        predictions = model_fit.transform(test_df)

        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_pipeline = model_fit
            best_model_name = name

    # Safety check
    if best_model_pipeline is None:
        print("❌ No model was successfully trained.")
        return

    print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.4f}")

    # Model saving - Windows-safe local FS
    try:
        model_dir = os.path.dirname(BEST_MODEL_PATH)
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir, exist_ok=True)

        model_path = f"file:///{os.path.abspath(BEST_MODEL_PATH)}"
        best_model_pipeline.write().overwrite().save(model_path)

        print(f"✅ Best model saved to: {BEST_MODEL_PATH} ({model_path})")

    except Exception as e:
        print(f"❌ Failed to save model: {e}")

    # Sklearn backup (Python 3.13 compat)
    print("Training sklearn RF backup...")
    df_sample = df.sample(False, 0.05, seed=42).toPandas()
    categorical = ["VendorID", "RatecodeID", "payment_type"]
    for c in categorical:
        df_sample[c] = df_sample[c].astype('category').cat.codes
    features = categorical + ["passenger_count", "trip_distance", "pickup_hour", "day_of_week", "trip_duration"]
    X = df_sample[features]
    y = df_sample["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Sklearn RF RMSE: {rmse:.4f}")
    sk_model_path = "models/sk_rf.pkl"
    os.makedirs(os.path.dirname(sk_model_path), exist_ok=True)
    joblib.dump(rf, sk_model_path)
    print(f"✅ Sklearn model saved to {sk_model_path}")


if __name__ == "__main__":
    train_models()
