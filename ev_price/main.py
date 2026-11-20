# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving the model and scaler
from sklearn.preprocessing import MinMaxScaler  # If scaling is required

# Set style
sns.set_context('notebook', font_scale=1.5)
color_graph = '#8A9A5B'
color_background = '#E5E4E2'

# Load data
filename = '../ev_price/Electric cars.csv'
df = pd.read_csv(filename)

# Display initial data to understand its structure
print("Initial Data:")
print(df.head())

# Clean data - Drop unwanted columns (e.g., "Unnamed")
unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
if unnamed_columns:
    df = df.drop(columns=unnamed_columns)

# Convert columns to numeric and handle errors
df_cleaned = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
df_cleaned = df_cleaned.dropna()

# Check if there are any rows left after cleaning
if df_cleaned.empty:
    raise ValueError("DataFrame is empty after cleaning. Ensure the dataset contains numeric values.")

# Display cleaned data
print("\nCleaned Data:")
print(df_cleaned.head())

# Generate synthetic data (create ~10,000 rows) with the cleaned data
np.random.seed(42)
synthetic_data = []

for _ in range(10000):
    row = df_cleaned.sample(1, replace=True).iloc[0]
    new_row = row.copy()

    # Add small random noise (within +-10%)
    for col in df_cleaned.columns:
        if col != 'Year' and np.issubdtype(df_cleaned[col].dtype, np.number):
            noise = np.random.uniform(0.9, 1.1)
            new_row[col] = new_row[col] * noise
    
    # Randomly assign year between 2010 and 2025
    new_row['Year'] = np.random.randint(2010, 2026)

    synthetic_data.append(new_row)

# Create DataFrame from synthetic data
synthetic_df = pd.DataFrame(synthetic_data)

# Format types
synthetic_df = synthetic_df.apply(pd.to_numeric, errors='ignore')
synthetic_df['Year'] = synthetic_df['Year'].astype(int)

# Set 'Year' as index
synthetic_df.set_index('Year', inplace=True)

# Define feature columns and target column
feature_cols = [col for col in synthetic_df.columns if col != 'Global Sales Volume']
target_col = 'Global Sales Volume'

X = synthetic_df[feature_cols]
y = synthetic_df[target_col]

# Scale the features (optional, but good for many models)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Regression Results (Synthetic Data) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# Compare actual vs predicted
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
print("\nActual vs Predicted:")
print(comparison_df.head(10))

# Visualization: Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(15, 8))
sns.barplot(x=[feature_cols[i] for i in indices], y=importances[indices], palette='viridis')
plt.title('Feature Importances', fontsize=20, color=color_graph)
plt.xticks(rotation=45)
plt.show()

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color=color_graph)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Global Sales Volume', fontsize=18, color=color_graph)
plt.show()

# Save the trained Random Forest model
model_filename = 'random_forest_model.joblib'
joblib.dump(rf, model_filename)
print(f"Model saved as {model_filename}")

# Save the scaler (if used)
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")
