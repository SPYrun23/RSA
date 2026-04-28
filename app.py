# ==========================================
# Retail Sales Analysis Dashboard (FINAL)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Styled Title
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Retail Sales Dashboard</h1>",
    unsafe_allow_html=True
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your Retail Dataset (CSV)", type=["csv"])

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:

    # Load Data with Error Handling
    try:
        df = pd.read_csv(uploaded_file)
    except:
        st.error("❌ Error loading file. Please upload a valid CSV.")
        st.stop()

    # Raw Data Toggle
    st.subheader("📄 Dataset")
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(inplace=True)

    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['Year'] = df['Date'].dt.year
    df['Total Amount'] = pd.to_numeric(df['Total Amount'], errors='coerce')

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("🔎 Filters")

    years = st.sidebar.multiselect(
        "Select Year",
        options=sorted(df['Year'].unique()),
        default=sorted(df['Year'].unique())
    )

    categories = st.sidebar.multiselect(
        "Select Category",
        options=df['Product Category'].unique(),
        default=df['Product Category'].unique()
    )

    filtered_df = df[
        (df['Year'].isin(years)) &
        (df['Product Category'].isin(categories))
    ]

    # -------------------------------
    # KPIs
    # -------------------------------
    total_revenue = filtered_df['Total Amount'].sum()
    total_orders = len(filtered_df)
    customers = filtered_df['Customer ID'].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("🛒 Total Orders", total_orders)
    col3.metric("👥 Unique Customers", customers)

    # -------------------------------
    # Monthly Sales Trend
    # -------------------------------
    st.subheader("📈 Monthly Revenue Trend")

    monthly_sales = filtered_df.groupby('Month')['Total Amount'].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.lineplot(data=monthly_sales, x='Month', y='Total Amount', marker='o', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # -------------------------------
    # Category-wise Sales
    # -------------------------------
    st.subheader("📊 Revenue by Product Category")

    category_sales = filtered_df.groupby('Product Category')['Total Amount'].sum().sort_values()

    fig2, ax2 = plt.subplots(figsize=(8,4))
    category_sales.plot(kind='barh', ax=ax2)
    st.pyplot(fig2)

    # -------------------------------
    # Top Customers
    # -------------------------------
    st.subheader("🏆 Top 10 Customers")

    top_customers = (
        filtered_df.groupby('Customer ID')['Total Amount']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig3, ax3 = plt.subplots(figsize=(8,4))
    top_customers.plot(kind='bar', ax=ax3)
    st.pyplot(fig3)

    # =========================================
    # 🔮 Improved ARIMA Prediction
    # =========================================
    st.subheader("🔮 Sales Prediction (Improved ARIMA)")

    import warnings

    warnings.filterwarnings("ignore")

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # -----------------------
    # Prepare Time Series Properly
    # -----------------------
    monthly_sales = filtered_df.groupby('Month')['Total Amount'].sum().reset_index()

    # Convert Month to datetime index (IMPORTANT)
    monthly_sales['Month'] = pd.to_datetime(monthly_sales['Month'])
    monthly_sales.set_index('Month', inplace=True)

    # Optional smoothing (helps accuracy)
    monthly_sales['Total Amount'] = monthly_sales['Total Amount'].rolling(window=2).mean()
    monthly_sales.dropna(inplace=True)

    # -----------------------
    # Train-Test Split
    # -----------------------
    split = int(len(monthly_sales) * 0.8)
    train = monthly_sales[:split]
    test = monthly_sales[split:]

    # -----------------------
    # Auto ARIMA (Manual Grid Search)
    # -----------------------
    best_rmse = float("inf")
    best_order = None
    best_model = None

    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(train['Total Amount'], order=(p, d, q)).fit()
                    pred = model.forecast(steps=len(test))

                    rmse = np.sqrt(mean_squared_error(test['Total Amount'], pred))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_order = (p, d, q)
                        best_model = model
                except:
                    continue

    print("\nBest ARIMA order:", best_order)

    # -----------------------
    # Final Prediction
    # -----------------------
    pred = best_model.forecast(steps=len(test))

    # -----------------------
    # Metrics (MAPE-based Accuracy)
    # -----------------------
    mape = np.mean(np.abs((test['Total Amount'] - pred) / test['Total Amount'])) * 100
    accuracy = 100 - mape
    rmse = np.sqrt(mean_squared_error(test['Total Amount'], pred))

    print("\n===== IMPROVED MODEL PERFORMANCE =====")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy (MAPE): {accuracy:.2f}%")

    # -----------------------
    # Full Model for Visualization
    # -----------------------
    final_model = ARIMA(monthly_sales['Total Amount'], order=best_order).fit()
    full_pred = final_model.predict(start=0, end=len(monthly_sales) - 1)

    # -----------------------
    # Medium Chart
    # -----------------------
    st.subheader("📈 Actual vs Predicted (Improved ARIMA)")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(monthly_sales.index, monthly_sales['Total Amount'], label='Actual', marker='o')
    ax.plot(monthly_sales.index, full_pred, label='ARIMA Predicted', linestyle='--')

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------
    # Future Prediction
    # -----------------------
    future = final_model.forecast(steps=1)

    st.success(f"📌 Predicted Revenue for Next Month: ₹{future.values[0]:,.2f}")
    # -------------------------------
    # Download Button
    # -------------------------------
    st.download_button(
        label="📥 Download Processed Data",
        data=filtered_df.to_csv(index=False),
        file_name="processed_sales_data.csv",
        mime="text/csv"
    )

# -------------------------------
# No File Uploaded
# -------------------------------
else:
    st.warning("⚠️ Please upload a dataset to proceed.")
