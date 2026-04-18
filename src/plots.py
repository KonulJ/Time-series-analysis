import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("ggplot")


def _pics_dir(src_dir: str) -> str:
    path = os.path.join(src_dir, "pics")
    os.makedirs(path, exist_ok=True)
    return path


def plot_all_eda(ts, src_dir: str) -> None:
    print("\n[2/4] Generating EDA charts...")
    out = _pics_dir(src_dir)

    plt.figure(figsize=(12, 5))
    plt.plot(ts, color="#5D9C59", label="Avg Price")
    plt.title("Historical Price Trend (Total US)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "1_trend.png"))
    plt.close()

    rolling_mean = ts.rolling(window=12).mean()
    rolling_std = ts.rolling(window=12).std()
    plt.figure(figsize=(12, 5))
    plt.plot(ts, alpha=0.3, label="Original")
    plt.plot(rolling_mean, color="red", label="Rolling Mean (12wk)")
    plt.plot(rolling_std, color="black", label="Volatility (Std)")
    plt.title("Rolling Mean & Stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "2_rolling.png"))
    plt.close()

    decomp = seasonal_decompose(ts, model="additive", period=52)
    fig = decomp.plot()
    fig.set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "3_decomposition.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(ts, kde=True, color="blue")
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "4_distribution.png"))
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(ts.diff().dropna(), ax=ax1, lags=40)
    ax1.set_title("Autocorrelation (ACF)")
    plot_pacf(ts.diff().dropna(), ax=ax2, lags=40)
    ax2.set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "5_acf_pacf.png"))
    plt.close()

    print(f"      Charts saved to {out}/")


def plot_final_forecast(train, test, forecast, conf_int, mape, src_dir: str) -> None:
    out = _pics_dir(src_dir)

    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label="Training Data", color="gray", alpha=0.5)
    plt.plot(test.index, test, label="Actual Price", color="green", linewidth=2)
    plt.plot(test.index, forecast, label="Forecast", color="red", linestyle="--", linewidth=2)
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
    plt.title(f"SARIMA Forecast | MAPE: {mape:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "6_final_forecast.png"))
    plt.close()
    print(f"      Forecast chart saved to {out}/6_final_forecast.png")
