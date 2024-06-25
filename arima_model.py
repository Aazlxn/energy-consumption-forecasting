import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

print("Loading data...")
# Load the data
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

print("Training ARIMA model...")
# Train ARIMA model
arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

print("Making predictions...")
# Make predictions
y_pred_arima = arima_model_fit.forecast(steps=len(y_test))

print("Evaluating model...")
# Evaluate model
rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
print(f'RMSE for ARIMA: {rmse_arima}')

# Save predictions
pd.DataFrame(y_pred_arima).to_csv('data/y_pred_arima.csv', index=False)
print("ARIMA model complete.")