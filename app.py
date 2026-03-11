import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Page Config
st.set_page_config(page_title="House Price Prediction AI", layout="wide")

# Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('house_price_model.pkl')

@st.cache_data
def load_data():
    df_raw = fetch_openml(name='house_prices', as_frame=True, parser='auto').frame
    cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'SalePrice']
    df = df_raw[cols].copy()
    df.rename(columns={'OverallQual': 'OverallQuality'}, inplace=True)
    df.dropna(inplace=True)
    return df

try:
    pipeline = load_pipeline()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}. Please ensure you have run main_program.py first.")
    st.stop()

# Build app presentation
st.title("PragyanAI - House Price Prediction AI")
st.markdown("Predict the **sale price of houses** using property attributes based on historical records.")

# 1. EDA Section
st.subheader("Exploratory Data Analysis (EDA)")
if st.checkbox("Show SalePrice Distribution"):
    st.write("Using seaborn to plot the distribution of SalePrice:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["SalePrice"], kde=True, ax=ax, color="blue", bins=50)
    ax.set_title("Distribution of SalePrice")
    ax.set_xlabel("Sale Price (Rs.)")
    st.pyplot(fig)
    
    # Also show basic summary
    st.write("**Data Summary:**")
    st.dataframe(df.describe())

st.divider()

# 2. Prediction Section
st.subheader("Predict House Price")
st.write("Enter the property attributes below to estimate the sale price.")

col1, col2, col3 = st.columns(3)
with col1:
    lot_area = st.number_input("LotArea (sq ft)", min_value=1300, max_value=250000, value=10000)
    overall_quality = st.slider("OverallQuality (1-10)", min_value=1, max_value=10, value=6)

with col2:
    year_built = st.number_input("YearBuilt", min_value=1870, max_value=2024, value=2000)
    total_bsmt_sf = st.number_input("TotalBsmtSF (sq ft)", min_value=0, max_value=6000, value=1000)

with col3:
    garage_cars = st.number_input("GarageCars", min_value=0, max_value=5, value=2)

st.markdown("---")

if st.button("Predict Sale Price", type="primary"):
    # Feature Engineering logic mimicking main_program.py
    # Using 2010 as Max YearBuilt from the original dataset source to represent Age
    age = 2010 - year_built 
    
    # Needs exactly matching feature order for sklearn prediction
    # Features trained: ['LotArea', 'OverallQuality', 'YearBuilt', 'TotalBsmtSF', 'GarageCars', 'Age']
    input_df = pd.DataFrame([{
        'LotArea': lot_area,
        'OverallQuality': overall_quality,
        'YearBuilt': year_built,
        'TotalBsmtSF': total_bsmt_sf,
        'GarageCars': garage_cars,
        'Age': age
    }])
    
    with st.spinner("Running AI Model Prediction..."):
        prediction = pipeline.predict(input_df)[0]
    
    st.success(f"### Estimated Sale Price: **Rs. {prediction:,.2f}**")
    
    # 3. Context Visualization
    st.write("### How this compares to the local market:")
    temp_df = df.copy()
    temp_df['Type'] = 'Historical Sales'
    
    input_viz = input_df.copy()
    input_viz['SalePrice'] = prediction
    input_viz['Type'] = 'Your Property'
    
    combined = pd.concat([temp_df, input_viz], ignore_index=True)
    
    fig = px.scatter(
        combined, x='LotArea', y='SalePrice', color='Type', 
        hover_data=['OverallQuality', 'YearBuilt'],
        title="Your Property compared to Historical Dataset",
        log_x=True, log_y=True,
        color_discrete_map={'Historical Sales': '#636EFA', 'Your Property': '#EF553B'}
    )
    
    # Enhance the point configuration
    fig.update_traces(marker=dict(size=7, opacity=0.6), selector=dict(name='Historical Sales'))
    fig.update_traces(marker=dict(size=18, symbol='star', opacity=1), selector=dict(name='Your Property'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show features used by model
    st.write("#### Engineered Features Used:")
    st.json({"Log Transform Target": "Handled Internally by TransformedTargetRegressor", "Age Feature (Computed as 2010 - YearBuilt)": age})
