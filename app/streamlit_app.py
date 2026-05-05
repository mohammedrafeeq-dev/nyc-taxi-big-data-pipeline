import sys
import streamlit as st
import os
import pandas as pd
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEST_MODEL_PATH

# Page Config
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #f63366;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model (sklearn fallback)."""
    sk_path = 'models/sk_rf.pkl'
    if os.path.exists(sk_path):
        try:
            rf = joblib.load(sk_path)
            return ('rf', rf)
        except Exception as e:
            st.error(f"Sklearn load error: {e}")
    return None

def main():
    st.title("🚕 NYC Taxi Trip Fare Prediction")
    st.markdown("---")
    
    st.sidebar.header("📍 Trip Details")
    
    # User Inputs
    vendor_id = st.sidebar.selectbox("Vendor ID", [1, 2], help="1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.")
    rate_code = st.sidebar.selectbox("Ratecode ID", [1, 2, 3, 4, 5, 6], help="1=Standard, 2=JFK, 3=Newark, etc.")
    passenger_count = st.sidebar.slider("Passenger Count", 1, 6, 1)
    trip_distance = st.sidebar.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=2.5)
    payment_type = st.sidebar.selectbox("Payment Type", [1, 2, 3, 4], help="1=Credit card, 2=Cash, 3=No charge, 4=Dispute")
    
    # Time Inputs
    pickup_hour = st.sidebar.slider("Pickup Hour", 0, 23, 12)
    day_of_week = st.sidebar.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3)
    trip_duration = st.sidebar.number_input("Estimated Duration (min)", min_value=1.0, max_value=180.0, value=15.0)

    # Load Model
    model = load_model()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Predict Fare Amount")
        st.info("Input the trip details in the sidebar to estimate the fare.")
        
        if st.sidebar.button("Predict Fare"):
            if model is None:
                st.error("Model not found! Please train the model first.")
            else:
                data = [vendor_id, rate_code, passenger_count, trip_distance, payment_type, pickup_hour, day_of_week, trip_duration]
                cols = ["VendorID", "RatecodeID", "passenger_count", "trip_distance", "payment_type", "pickup_hour", "day_of_week", "trip_duration"]
                df = pd.DataFrame([data], columns=cols)
                categorical = ["VendorID", "RatecodeID", "payment_type"]
                for c in categorical:
                    df[c] = df[c].astype('category').cat.codes
                features = categorical + ["passenger_count", "trip_distance", "pickup_hour", "day_of_week", "trip_duration"]
                X = df[features]
                fare = model[1].predict(X)[0]
                
                # Display Results
                st.metric(label="Estimated Fare Amount", value=f"${fare:.2f}")
                
                st.success("Prediction complete!")
                
                # Show input summary
                st.write("### Trip Summary")
                st.table(pd.DataFrame(data, columns=cols))
    
    with col2:
        st.subheader("Model Insights")
        if os.path.exists("plots/fare_distribution.png"):
            st.image("plots/fare_distribution.png", caption="Historical Fare Distribution")
        if os.path.exists("plots/distance_vs_fare.png"):
            st.image("plots/distance_vs_fare.png", caption="Distance vs Fare Analysis")

if __name__ == "__main__":
    main()
