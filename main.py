import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Streamlit configuration
st.set_page_config(page_title="Sales Prediction Tool", layout="wide")

# App title
st.title("Sales Prediction Tool")

# File Upload
uploaded_file = st.file_uploader("Upload your sales data file (CSV format):", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Allow user to select the target column
    target_column = st.selectbox("Select the target column (e.g., Sales):", options=data.columns)
    if target_column:
        st.write(f"You selected the target column: **{target_column}**")

        # Handle preprocessing
        # Convert dates if a 'Date' column exists
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.sort_values('Date')
            st.write("Processed 'Date' column.")
        
        # Encode categorical variables dynamically
        st.write("Processing categorical columns...")
        for col in data.select_dtypes(include=['object']).columns:
            if col != target_column:  # Don't encode the target column
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                st.write(f"Encoded column: **{col}**")

        # Drop rows with NaN in the target column or non-numeric features
        data = data.dropna(subset=[target_column])
        X = data.drop(columns=[target_column], errors='ignore')
        y = data[target_column]

        # Ensure X contains only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_type = st.selectbox("Select a model type:", ["Linear Regression", "Random Forest", "Time Series (Holt-Winters)"])

        if model_type == "Linear Regression":
            st.subheader("Linear Regression Model")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_predictions = lr.predict(X_test)

            mse = mean_squared_error(y_test, lr_predictions)
            r2 = r2_score(y_test, lr_predictions)

            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R2 Score: {r2}")

        elif model_type == "Random Forest":
            st.subheader("Random Forest Model")
            rfr = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr.fit(X_train, y_train)
            rfr_predictions = rfr.predict(X_test)

            mse = mean_squared_error(y_test, rfr_predictions)
            r2 = r2_score(y_test, rfr_predictions)

            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R2 Score: {r2}")

        elif model_type == "Time Series (Holt-Winters)":
            st.subheader("Time Series Forecasting (Holt-Winters)")
            if 'Date' in data.columns:
                sales_ts = data.set_index('Date')[target_column]
                train_ts = sales_ts[:-12]  # Training data (all except the last 12 months)
                test_ts = sales_ts[-12:]   # Testing data (last 12 months)

                model = ExponentialSmoothing(train_ts, seasonal='add', seasonal_periods=12)
                hw_model = model.fit()
                hw_predictions = hw_model.forecast(12)

                st.write("Holt-Winters Predictions:")
                st.write(hw_predictions)

                # Plot actual vs. predicted
                st.line_chart(pd.DataFrame({"Actual": sales_ts, "Predicted": hw_predictions}))

                mse = mean_squared_error(test_ts, hw_predictions)
                r2 = r2_score(test_ts, hw_predictions)

                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R2 Score: {r2}")
            else:
                st.error("Time series model requires a 'Date' column in the dataset.")

        # Save the models
        save_model = st.checkbox("Save the model to disk?")
        if save_model:
            import joblib
            if model_type == "Linear Regression":
                joblib.dump(lr, 'linear_regression_model.pkl')
            elif model_type == "Random Forest":
                joblib.dump(rfr, 'random_forest_model.pkl')
            elif model_type == "Time Series (Holt-Winters)":
                joblib.dump(hw_model, 'holt_winters_model.pkl')

            st.success("Model saved successfully.")

    else:
        st.error("Please select a valid target column.")
else:
    st.info("Please upload a CSV file to begin.")
