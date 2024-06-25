# Energy Consumption Forecasting

## Project Overview

This project aims to forecast future energy consumption using historical data from 1985 to 2023. The primary goal is to develop models that can accurately predict future energy usage, which can be beneficial for energy planning, resource allocation, cost management, and policy-making.

We compare the performance of two models: ARIMA and LSTM. The project involves data preparation, model development, evaluation, and forecasting future energy consumption.

## Data Preparation

The historical data includes total annual sales (in GWh) and the number of customers by market segment (Residential, Farm, Commercial, and Industrial) from 1985 to 2023. The data preparation steps involved:

1. **Loading Data**: The data was loaded from an Excel file.
2. **Renaming and Cleaning Columns**: Columns were renamed for better readability, and non-numeric values were handled.
3. **Handling Non-numeric Values**: Rows with non-numeric values in the 'Year' column were removed.
4. **Converting Data Types**: Data types were converted to appropriate formats.
5. **Splitting Data**: The data was split into training (1985-2017) and testing (2018-2023) sets.

## Model Development

### ARIMA Model

The ARIMA (AutoRegressive Integrated Moving Average) model was trained on the training data. The optimal parameters for the ARIMA model were chosen using the `auto_arima` function, which automates the process of finding the best parameters.

- **Optimal Parameters**: The optimal order for the ARIMA model was determined to be `(5, 1, 0)`.
- **RMSE**: The Root Mean Square Error (RMSE) for the ARIMA model on the test set was approximately 45185.44 GWh.
- **Findings**: The ARIMA model showed stable forecasts with minimal variability, which may not fully capture the trends and fluctuations in the data.

### LSTM Model

The LSTM (Long Short-Term Memory) model was trained on the normalized and reshaped training data. LSTM models are well-suited for time series forecasting due to their ability to capture long-term dependencies.

- **Normalization**: The data was normalized to a range between 0 and 1.
- **Reshaping Data**: The data was reshaped into the format required by the LSTM model.
- **Training**: The LSTM model was trained with 50 units and 200 epochs.
- **RMSE**: The RMSE for the LSTM model on the test set was approximately 10407.95 GWh.
- **Findings**: The LSTM model showed more variability in the predictions, which is more realistic for energy consumption data. However, some repeated values suggested there might be issues with the input data handling.

## Future Predictions

Using the LSTM model, future energy consumption for the next 5 years was forecasted. The predicted values show a decreasing trend, which will be analyzed further.

- **Predicted Values for Future Years**:
  - Year 33: 52867.8 GWh
  - Year 34: 51479.37 GWh
  - Year 35: 50514.42 GWh
  - Year 36: 49852.59 GWh
  - Year 37: 49402.824 GWh

## Results

### ARIMA Model Predictions
- **RMSE**: Approximately 45185.44 GWh.
- **Analysis**: The ARIMA model provided stable forecasts with minimal variability. This stability indicates that the model might not be capturing the more subtle trends and fluctuations present in the data.

### LSTM Model Predictions
- **RMSE**: Approximately 10407.95 GWh.
- **Analysis**: The LSTM model showed variability in the predictions, which is more reflective of real-world energy consumption patterns. However, repeated values in the predictions suggest issues with data input handling that need to be addressed.

## Conclusion

This project demonstrates the use of ARIMA and LSTM models for forecasting energy consumption. Both models have their strengths and weaknesses:
- The ARIMA model is straightforward and provides stable forecasts but might not capture all trends.
- The LSTM model, while more complex and capable of capturing variability, requires careful handling of data inputs and parameters.

## Next Steps

- **Validate LSTM Model Predictions**: Address the issues with repeated values and refine the input data handling.
- **Refine Future Forecasts**: Ensure the future forecasts align with historical trends and expected future scenarios.
- **Document Findings**: Continue documenting the process and findings for ongoing learning and improvement.

## How to Run the Project

1. Clone the repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Run the data preperation script:
   python data_preparation.py
4. Train the models and make predictions: 
   python arima_model.py
   python lstm_model.py
