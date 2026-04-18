# Avocado Price Forecasting — Time Series Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![SARIMA](https://img.shields.io/badge/Model-SARIMA-green)
![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Forecasts weekly conventional avocado prices across the US (2015–2018) using SARIMA. Structured as a modular Python pipeline, not a notebook — each stage is a separate module with a single entry point.

---

## Model Performance

| Metric | Score | Interpretation |
| --- | --- | --- |
| RMSE | 0.22 | ±$0.22 average error |
| MAE | 0.18 | Median absolute deviation |
| MAPE | 13.58% | ~86.5% directional accuracy |

---

## Project Structure

```text
Time-series-analysis/
├── data/
│   └── avocado.csv          # Raw dataset (Kaggle — not committed)
├── src/
│   ├── main.py              # Entry point — runs full pipeline
│   ├── loader.py            # Data ingestion and preprocessing
│   ├── model.py             # ADF stationarity test + SARIMA training
│   ├── plots.py             # EDA and forecast visualisations
│   └── pics/                # Auto-generated charts (git-ignored)
├── notebooks/
│   └── exploration.ipynb    # EDA sandbox
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/KonulJ/Time-series-analysis.git
cd Time-series-analysis
pip install -r requirements.txt
```

Download `avocado.csv` from [Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices) and place it in `data/`.

---

## Run

```bash
python src/main.py
```

The pipeline runs four stages in sequence:

1. Load and preprocess — filter TotalUS conventional, resample to weekly, interpolate gaps
2. EDA — generate 5 charts (trend, rolling stats, decomposition, distribution, ACF/PACF)
3. Stationarity — ADF test to confirm differencing requirement
4. SARIMA — 80/20 train/test split, fit `(1,1,1)x(1,1,1,52)`, evaluate, save forecast chart

All charts are saved to `src/pics/`.

---

## Key Concepts Demonstrated

| Concept | Implementation |
| --- | --- |
| Time series preprocessing | Resampling, interpolation, stationarity testing (ADF) |
| Seasonal decomposition | `statsmodels.tsa.seasonal.seasonal_decompose` — trend + seasonality + residual |
| SARIMA modelling | `SARIMAX(1,1,1)(1,1,1,52)` — weekly seasonality captured |
| Forecast evaluation | RMSE, MAE, MAPE with 95% confidence intervals |
| Modular pipeline | No notebooks in production path — clean `src/` with single entry point |

---

## Roadmap

- [ ] Prophet and LSTM comparison
- [ ] Auto hyperparameter tuning (auto-arima)
- [ ] Streamlit dashboard for interactive forecasting

---

## Data

[Avocado Prices — Kaggle](https://www.kaggle.com/datasets/neuromusic/avocado-prices) | Hass Avocado Board, 2018

---

*Built by [Konul Jafarova](https://github.com/KonulJ)*
