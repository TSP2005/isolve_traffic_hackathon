import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load the dataset
file_path = 'Electronic_City_Phase_1_Traffic_Data_with_Accident_Scale_2023.csv'
traffic_data = pd.read_csv(file_path)

# Preprocessing
df = traffic_data.copy()

# Convert 'Time' to minutes from midnight
df['Time'] = pd.to_datetime(df['Time']).dt.hour * 60 + pd.to_datetime(df['Time']).dt.minute

# Encode categorical variables
label_encoders = {}
for column in ['Weekday', 'Lane', 'Weather', 'Public_Holiday']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# -------------- Selecting the Test Set based on Time --------------

# Select 5 distinct times and get all lanes for those times
distinct_times = df['Time'].unique()[:5]  # Get 5 distinct time points
test_set = df[df['Time'].isin(distinct_times)]  # Select rows with those times
train_set = df[~df['Time'].isin(distinct_times)]  # Exclude those times from the training set

# Features and target for training set
X_train = train_set[['Vehicle_Count', 'Weekday', 'Time', 'Lane', 'Weather', 'Public_Holiday', 'Accident_Scale']]
y_train = train_set['Congestion_Level']

# Test set features and target
X_test = test_set[['Vehicle_Count', 'Weekday', 'Time', 'Lane', 'Weather', 'Public_Holiday', 'Accident_Scale']]
y_test = test_set['Congestion_Level']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple ANN model (or use any other regression model in place of ANN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build a simple ANN model
ann_model = Sequential()
ann_model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
ann_model.add(Dense(units=64, activation='relu'))
ann_model.add(Dense(units=1))  # Output layer

# Compile the model
ann_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Predict on the test set
y_pred_ann = ann_model.predict(X_test_scaled)

# Flatten the predictions for comparison
y_pred_ann_flat = np.ravel(y_pred_ann)

# Evaluate the model
mse_ann = mean_squared_error(y_test, y_pred_ann_flat)
r2_ann = r2_score(y_test, y_pred_ann_flat)

# Compare actual and predicted congestion levels (sample)
comparison_df = pd.DataFrame({'Actual Congestion Level': y_test, 'Predicted Congestion Level': y_pred_ann_flat})
print(comparison_df.head())

# Calculate additional metrics like R-squared
print("ANN Mean Squared Error:", mse_ann)
print("ANN R-squared:", r2_ann)

# -------------- Predicting Signal Times using Linear and Polynomial Regression --------------

# For signal time prediction, train on the whole dataset
X_full = df[['Congestion_Level', 'Lane']]
y_full = df['Signal_Time']

# Train a Linear Regression model on the full dataset
linear_model = LinearRegression()
linear_model.fit(X_full, y_full)

# For testing, use predicted congestion levels from the test set
X_signal_test = pd.DataFrame({'Congestion_Level': y_pred_ann_flat, 'Lane': test_set['Lane']})

# Predict signal times using the predicted congestion levels
y_pred_signal_linear = linear_model.predict(X_signal_test)

# -------------- Polynomial Regression for Signal Time Prediction --------------

# Polynomial regression (degree 2) on the full dataset
poly = PolynomialFeatures(degree=2)
X_poly_full = poly.fit_transform(X_full)
poly_model = LinearRegression()
poly_model.fit(X_poly_full, y_full)

# Predict signal times using the polynomial model
X_poly_signal_test = poly.transform(X_signal_test)
y_pred_signal_poly = poly_model.predict(X_poly_signal_test)

# Evaluate and compare the signal time predictions with the original signal times in the test set
mse_signal_linear = mean_squared_error(test_set['Signal_Time'], y_pred_signal_linear)
mse_signal_poly = mean_squared_error(test_set['Signal_Time'], y_pred_signal_poly)

# Comparison of actual and predicted signal times
comparison_times_df = pd.DataFrame({
    'Time': test_set['Time'],
    'Lane': test_set['Lane'],
    'Actual Signal Time': test_set['Signal_Time'],
    'Predicted Signal Time (Linear)': y_pred_signal_linear,
    'Predicted Signal Time (Polynomial)': y_pred_signal_poly
})

# Group the results by time to compare signal times for all lanes at the same time
comparison_grouped = comparison_times_df.groupby('Time')

# Show a few rows of the grouped comparison
for time, group in comparison_grouped:
    print(f"\nTime: {time}")
    print(group[['Lane', 'Actual Signal Time', 'Predicted Signal Time (Linear)', 'Predicted Signal Time (Polynomial)']])

print(f"Linear Regression Mean Squared Error for Signal Time: {mse_signal_linear}")
print(f"Polynomial Regression Mean Squared Error for Signal Time: {mse_signal_poly}")

# Optionally, save the results to a CSV
comparison_times_df.to_csv('signal_time_comparison_grouped.csv', index=False)
