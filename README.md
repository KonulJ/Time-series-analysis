# 🥑 Avocado Price Forecasting System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Model](https://img.shields.io/badge/Model-SARIMA-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

> A production-ready Time Series application that forecasts the average price of conventional avocados in the US using SARIMA.

---

## ⭐ Features
- Fully automated time-series pipeline
- Modular project architecture
- Statistical validation using ADF Test
- Automatic visualization generation
- Forecast evaluation with multiple metrics
- Production-style repository structure

---

## 📖 Project Overview

This project analyzes historical avocado prices (2015–2018) to predict future market trends using **SARIMA time-series modeling**.

The project is structured as a **modular Python application**, not a notebook.

### 🎯 Objectives
- Analyze market trend and seasonality
- Test stationarity using the ADF test
- Train a SARIMA forecasting model
- Generate automated visual reports

---

## 📂 Project Structure

```text

```text
avocado-sales/
│
├── data/
│   └── avocado.csv          # Raw dataset (Kaggle)
│
├── src/                     # Application Source Code
│   ├── main.py              # 🚀 Entry point
│   ├── loader.py            # Data cleaning & preprocessing
│   ├── model.py             # Statistical modeling (SARIMA)
│   ├── plots.py             # Visualization engine
│   └── pics/                # Generated charts (saved automatically)
│
├
├── requirements.txt
└── README.md
```


---

## 🚀 How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline
```bash
cd src
python main.py
```

The script will:
- Load data
- Train SARIMA model
- Print metrics
- Generate charts automatically

---


## 📊 Model Performance

| Metric | Score | Interpretation |
|---|---|---|
| RMSE | 0.22 | ±22 cents error |
| MAE | 0.18 | Avg absolute error |
| MAPE | 13.58% | ~86.5% accuracy |

---

## 🛠 Tech Stack
- Pandas
- Statsmodels
- Matplotlib / Seaborn
- Scikit-Learn

---

## 💡 Future Improvements
- Prophet & LSTM comparison
- Streamlit deployment
- Auto hyperparameter tuning

---

## 👩‍💻 Author
**Konul Jafarova**
