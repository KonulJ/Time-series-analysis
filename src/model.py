from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def check_stationarity(ts):
    result = adfuller(ts)
    print(f"\n      ADF Statistic: {result[0]:.4f}")
    print(f"      p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("      Result: Stationary")
    else:
        print("      Result: Non-Stationary (Differencing required)")

def train_and_forecast(ts):
    print("\n[3/4] Training SARIMA Model...")
    
    # 1. Split (80/20)
    train_size = int(len(ts) * 0.8)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]
    
    # 2. Define Model (1,1,1) x (1,1,1,52)
    # Seasonal period = 52 (Weekly data)
    model = SARIMAX(train, 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 52),
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    
    # 3. Fit
    results = model.fit(disp=False)
    print("      Model Trained Successfully.")
    
    # 4. Forecast
    forecast_res = results.get_forecast(steps=len(test))
    pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    
    # 5. Metrics
    rmse = np.sqrt(mean_squared_error(test, pred_mean))
    mae = mean_absolute_error(test, pred_mean)
    mape = np.mean(np.abs((test - pred_mean) / test)) * 100
    
    print(f"\n[4/4] Model Evaluation:")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      MAE:  {mae:.4f}")
    print(f"      MAPE: {mape:.2f}%")
    
    return train, test, pred_mean, conf_int, mape