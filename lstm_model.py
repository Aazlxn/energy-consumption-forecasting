import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("Loading data...")
# Load the data
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

print("Normalizing data...")
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# Reshape data
print("Reshaping data...")
X_train_lstm = y_train_scaled.reshape((y_train_scaled.shape[0], 1, 1))
X_test_lstm = y_test_scaled.reshape((y_test_scaled.shape[0], 1, 1))

# Build LSTM model
print("Building LSTM model...")
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Train LSTM model
print("Training LSTM model...")
lstm_model.fit(X_train_lstm, y_train_scaled, epochs=200, verbose=0)

# Make predictions
print("Making predictions...")
y_pred_lstm_scaled = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)

# Evaluate model
print("Evaluating model...")
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
print(f'RMSE for LSTM: {rmse_lstm}')

# Save predictions
pd.DataFrame(y_pred_lstm).to_csv('data/y_pred_lstm.csv', index=False)

# Future predictions
print("Generating future predictions...")
future_years = 5
future_years_array = np.arange(y_train.index.max() + 1, y_train.index.max() + 1 + future_years)

# Initialize future predictions array
future_predictions = []

# Use the last available data point to start the predictions
last_data_point = y_train_scaled[-1]

for _ in range(future_years):
    # Reshape the last data point to match LSTM input shape
    last_data_point = last_data_point.reshape((1, 1, 1))

    # Predict the next value
    next_pred_scaled = lstm_model.predict(last_data_point)

    # Append the predicted value
    future_predictions.append(next_pred_scaled[0][0])

    # Update the last data point with the predicted value
    last_data_point = next_pred_scaled

# Inverse transform the predictions to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Combine the future years and their corresponding predictions
future_forecasts = pd.DataFrame({
    'Year': future_years_array,
    'Total_GWh': future_predictions.flatten()
})

# Save future predictions
future_forecasts.to_csv('data/future_forecasts.csv', index=False)
print(future_forecasts)
print("LSTM model complete.")