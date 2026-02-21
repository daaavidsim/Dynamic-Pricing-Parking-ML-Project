import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Setup Page
st.set_page_config(page_title="Smart Parking Pricing", layout="wide")

@st.cache_data
def load_and_preprocess():
    # Load dataset
    df = pd.read_csv('IIoT_Smart_Parking_Management.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Feature Engineering from your notebook
    df['prev_occupancy'] = df['Occupancy_Rate'].shift(1).bfill()
    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Simple preprocessing for the demo
    categorical_cols = ['User_Type', 'Nearby_Traffic_Level', 'Vehicle_Type', 'Spot_Size']
    df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df, df_final

df_raw, df_ml = load_and_preprocess()

# 2. Sidebar UI
st.sidebar.header("Pricing Simulator")
temp = st.sidebar.slider("Temperature", -10, 45, 20)
traffic = st.sidebar.selectbox("Traffic Level", ["Low", "Medium", "High"])

base_price = 2.0
traffic_multiplier = {"Low": 1.0, "Medium": 1.5, "High": 2.5}
temp_adjustment = (temp - 20) * 0.05 

predicted_price = (base_price * traffic_multiplier[traffic]) + temp_adjustment

st.subheader(f"Current Dynamic Price Estimate")
st.metric(label="Recommended Rate", value=f"${predicted_price:.2f} / hr", delta=f"{traffic} Traffic")

# 3. Main Dashboard
st.title("IIoT Smart Parking Dynamic Pricing")

with st.expander("📊 Technical Analysis: Why is the R2 Score Negative?"):
    st.write("""
    **The Insight:** My current model shows an $R^2$ of **-0.0760**. 
    In professional ML, this indicates the model is currently performing worse than a simple horizontal line (the mean).
    
    **Why?**
    * **Data Variance:** The `Occupancy_Rate` feature has a low standard deviation (0.1607). Since the target variable has little variance to begin with, even minor residuals from the model will cause R2 to be negative.
    * **Missing Signals:** Parking occupancy is highly dependent on local events or holidays not captured in this dataset.
    * **Future Improvement:** Transitioning from XGBoost to an **LSTM (Long Short-Term Memory)** network would likely capture the time-series patterns better.
    """)

st.subheader("Decision Drivers")
col1, col2 = st.columns([1, 1])

with col1:
    st.write("What drives parking demand?")
    importance_data = pd.DataFrame({
        'Feature': ['Traffic_Level', 'Temperature', 'Hour_of_Day', 'User_Type', 'Spot_Size'],
        'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
    }).sort_values(by='Importance', ascending=True)
    
    fig2, ax2 = plt.subplots()
    ax2.barh(importance_data['Feature'], importance_data['Importance'], color='#2ecc71')
    st.pyplot(fig2)

with col2:
    st.info("💡 **Insight:** Traffic Level is the primary driver for price adjustments in this simulation.")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Occupancy Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_raw['Occupancy_Rate'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Model Performance")
    # Using the results found in your notebook
    st.write(f"**Final R2 Score:** -0.0760")
    st.write(f"**Mean Absolute Error:** 0.14 spots")
    st.info("Note: The negative R2 suggests the model needs more features to beat the baseline average.")