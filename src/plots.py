import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set global style
plt.style.use('ggplot')

def plot_all_eda(ts):
    print("\n[2/4] Generating EDA Charts...")
    
    # GRAPH 1: Trend
    plt.figure(figsize=(12, 5))
    plt.plot(ts, color='#5D9C59', label='Avg Price')
    plt.title('1. Historical Price Trend (Total US)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1_trend.png')
    plt.close()

    # GRAPH 2: Rolling Statistics
    rolling_mean = ts.rolling(window=12).mean()
    rolling_std = ts.rolling(window=12).std()
    plt.figure(figsize=(12, 5))
    plt.plot(ts, alpha=0.3, label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean (12wk)')
    plt.plot(rolling_std, color='black', label='Volatility (Std)')
    plt.title('2. Rolling Mean & Stability')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2_rolling.png')
    plt.close()

    # GRAPH 3: Decomposition
    decomp = seasonal_decompose(ts, model='additive', period=52)
    fig = decomp.plot()
    fig.set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig('3_decomposition.png')
    plt.close()

    # GRAPH 4: Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(ts, kde=True, color='blue')
    plt.title('4. Price Distribution')
    plt.tight_layout()
    plt.savefig('4_distribution.png')
    plt.close()

    # GRAPHS 5 & 6: ACF / PACF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(ts.diff().dropna(), ax=ax1, lags=40)
    ax1.set_title('5. Autocorrelation (ACF)')
    plot_pacf(ts.diff().dropna(), ax=ax2, lags=40)
    ax2.set_title('6. Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.savefig('5_acf_pacf.png')
    plt.close()
    
    print("      Charts saved: 1_trend.png, 2_rolling.png, 3_decomposition.png, etc.")

def plot_final_forecast(train, test, forecast, conf_int, mape):
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data', color='gray', alpha=0.5)
    plt.plot(test.index, test, label='Actual Price', color='green', linewidth=2)
    plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--', linewidth=2)
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    
    plt.title(f'Final Forecast (SARIMA) | MAPE Error: {mape:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('6_final_forecast.png')
    plt.close()
    print("      Final Forecast Chart saved as '6_final_forecast.png'")