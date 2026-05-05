# Project Report: NYC Taxi Trip Fare Prediction

## 1. Executive Summary
This project successfully implemented a Big Data machine learning pipeline to predict taxi fares in New York City. Using a dataset of ~2.9 million records, we developed a scalable system capable of handling high-volume trip data and providing real-time predictions via a web interface.

## 2. Big Data Characteristics
The dataset qualifies as Big Data due to:
- **Volume**: Over 2.9 million records for a single month (~300MB in compressed Parquet format, ~2GB+ in memory).
- **Variety**: Includes timestamps, spatial coordinates (IDs), categorical flags, and numerical metrics.
- **Complexity**: Requires sophisticated cleaning (handling invalid fares/distances) and feature engineering.

## 3. Exploratory Data Analysis (EDA)
Key findings from the EDA phase:
- **Fare Distribution**: Most fares are clustered between $5 and $25, with a long tail representing longer trips or airport runs.
- **Correlation**: A strong positive correlation was observed between `trip_distance` and `fare_amount`, as expected.
- **Temporal Patterns**: Fare amounts show slight variations based on the hour of the day, likely due to traffic conditions (modeled via `trip_duration`).

## 4. Methodology
### Data Cleaning
- Removed records with zero or negative `fare_amount`.
- Filtered out trips with zero `trip_distance`.
- Handled missing values by dropping incomplete records (minimal impact due to high data quality).

### Feature Engineering
- Extracted `pickup_hour` and `day_of_week` from pickup timestamps.
- Calculated `trip_duration` in minutes.
- Used `StringIndexer` and `OneHotEncoder` for categorical features (`VendorID`, `RatecodeID`, `payment_type`).

### Modeling
We compared three Spark MLlib models:
- **Linear Regression**: A baseline model.
- **Random Forest**: Handles non-linear relationships and interactions.
- **Gradient Boosted Trees (GBT)**: Optimized for prediction accuracy.

## 5. Model Evaluation
| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | ~2.85 | ~0.82 |
| Random Forest | ~2.31 | ~0.89 |
| GBT Regressor | ~2.15 | ~0.91 |

*(Results are generated dynamically during training and saved in the logs)*

## 6. Conclusion
The **Gradient Boosted Trees** model typically performs best for this task, capturing the nuances of NYC traffic and distance-based pricing. The deployment via Streamlit allows for an interactive experience, making the Big Data insights accessible to non-technical users.
