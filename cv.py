import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'Electronic_City_Phase_1_Traffic_Data_with_Accident_Scale_2023.csv'  # Update the path accordingly
traffic_data = pd.read_csv(file_path)

# Preprocessing
df = traffic_data.copy()

# Convert 'Time' to minutes from midnight
df['Time'] = pd.to_datetime(df['Time']).dt.hour * 60 + pd.to_datetime(df['Time']).dt.minute

# Encode categorical variables for the entire dataset before splitting
label_encoders = {}
for column in ['Weekday', 'Lane', 'Weather', 'Public_Holiday']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# -------------- First Part: Predicting Lane Densities using ANN --------------

# Define features (excluding Signal_Time) and target (Congestion_Level)
X = df[['Vehicle_Count', 'Weekday', 'Time', 'Lane', 'Weather', 'Public_Holiday', 'Accident_Scale']]
y = df['Congestion_Level']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a simple ANN model
ann_model = Sequential()

# Input layer and hidden layers
ann_model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))
ann_model.add(Dense(units=64, activation='relu'))

# Output layer (1 neuron for congestion level prediction)
ann_model.add(Dense(units=1))

# Compile the model
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict on the test set
y_pred_ann = ann_model.predict(X_test)

# Flatten the predictions for comparison
y_pred_ann_flat = np.ravel(y_pred_ann)

# Evaluate the model
mse_ann = mean_squared_error(y_test, y_pred_ann_flat)
print("ANN Mean Squared Error:", mse_ann)

# Compare actual and predicted congestion levels (sample)
comparison_df = pd.DataFrame({'Actual Congestion Level': y_test, 'Predicted Congestion Level': y_pred_ann_flat})
print(comparison_df.head())

# Calculate additional metrics like R-squared
r2_ann = r2_score(y_test, y_pred_ann_flat)
print("ANN R-squared:", r2_ann)

# -------------- Second Part: Predicting Signal Times using Linear and Polynomial Regression --------------

# Use predicted congestion levels for the signal time prediction task
lane_density_features = df[['Congestion_Level', 'Lane']]
lane_signal_target = df['Signal_Time']

# Split into training and testing sets for signal time prediction
X_train_signal, X_test_signal, y_train_signal, y_test_signal = train_test_split(
    lane_density_features, lane_signal_target, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_signal, y_train_signal)
y_pred_linear = linear_model.predict(X_test_signal)

# Evaluate Linear Regression
mse_linear = mean_squared_error(y_test_signal, y_pred_linear)
r2_linear = r2_score(y_test_signal, y_pred_linear)

print(f"Linear Regression Mean Squared Error: {mse_linear}")
print(f"Linear Regression R-squared: {r2_linear}")

# 2. Polynomial Regression Model (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_signal)
X_poly_test = poly.transform(X_test_signal)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train_signal)
y_pred_poly = poly_model.predict(X_poly_test)

# Evaluate Polynomial Regression
mse_poly = mean_squared_error(y_test_signal, y_pred_poly)
r2_poly = r2_score(y_test_signal, y_pred_poly)

print(f"Polynomial Regression Mean Squared Error: {mse_poly}")
print(f"Polynomial Regression R-squared: {r2_poly}")

# -------------- Comparison of Actual and Predicted Signal Times --------------

# Create a comparison DataFrame for actual and predicted times
comparison_times_df = pd.DataFrame({
    'Actual Signal Time': y_test_signal,
    'Predicted Signal Time (Linear)': y_pred_linear,
    'Predicted Signal Time (Polynomial)': y_pred_poly
})

# Show a few rows of the comparison
print(comparison_times_df.head())

# Optionally, save to CSV if you want to analyze outside the notebook
comparison_times_df.to_csv('signal_time_comparison.csv', index=False)