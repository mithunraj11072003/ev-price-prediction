import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set style for visualization
sns.set_context('notebook', font_scale=1.5)
color_graph = '#8A9A5B'
color_background = '#F5F5F5'

# Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit UI
st.title('Electric vehicle Price Prediction')
st.markdown("""
This app allows you to upload a dataset of electric vehicle and predicts their sales volume using a trained Random Forest model.
The app will also visualize the actual vs predicted results and feature importance.
""")

# File upload widget
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the uploaded dataset
    test_df = pd.read_csv(uploaded_file)

    # Display the dataset
    st.subheader('Uploaded Dataset:')
    st.write(test_df.head())

    # Clean the data - Drop unwanted columns (e.g., "Unnamed")
    unnamed_columns = [col for col in test_df.columns if 'Unnamed' in col]
    if unnamed_columns:
        test_df = test_df.drop(columns=unnamed_columns)

    # Convert columns to numeric and handle errors
    test_df_cleaned = test_df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    test_df_cleaned = test_df_cleaned.dropna()

    # Check if there are any rows left after cleaning
    if test_df_cleaned.empty:
        st.error("DataFrame is empty after cleaning. Ensure the dataset contains numeric values.")
    else:
        st.subheader('Cleaned Data:')
        st.write(test_df_cleaned.head())

        # Define feature columns and target column
        feature_cols = [col for col in test_df_cleaned.columns if col != 'Global Sales Volume' and col != 'Year']
        target_col = 'Global Sales Volume'

        X_test = test_df_cleaned[feature_cols]
        y_test = test_df_cleaned[target_col]

        # Scale the features using the same scaler used during training
        X_test_scaled = scaler.transform(X_test)

        # Predict using the trained model
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model's performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader('Prediction Results:')
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2 Score): {r2:.4f}")

        # Compare actual vs predicted values
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.subheader('Actual vs Predicted Global Sales Volume:')
        st.write(comparison_df.head(10))

        # Visualization: Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(15, 8))
        sns.barplot(x=[feature_cols[i] for i in indices], y=importances[indices], palette='viridis')
        plt.title('Feature Importances', fontsize=20, color=color_graph)
        plt.xticks(rotation=45)
        st.subheader('Feature Importances:')
        st.pyplot(plt)

        # Visualization: Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color=color_graph)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Global Sales Volume', fontsize=18, color=color_graph)
        st.subheader('Actual vs Predicted Graph:')
        st.pyplot(plt)

        # Visualization: Residuals (Prediction errors)
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, color=color_graph)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values', fontsize=18, color=color_graph)
        st.subheader('Residuals vs Predicted Graph:')
        st.pyplot(plt)

        # Visualization: Distribution of Residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color=color_graph)
        plt.title('Distribution of Residuals', fontsize=18, color=color_graph)
        st.subheader('Residuals Distribution:')
        st.pyplot(plt)
