import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_and_preprocess
from plots import plot_all_eda, plot_final_forecast
from model import check_stationarity, train_and_forecast

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(src_dir), "data", "avocado.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: avocado.csv not found at {data_path}")
        sys.exit(1)

    ts = load_and_preprocess(data_path)
    plot_all_eda(ts, src_dir)
    check_stationarity(ts)
    train, test, forecast, conf_int, mape = train_and_forecast(ts)
    plot_final_forecast(train, test, forecast, conf_int, mape, src_dir)
    print(f"\nAnalysis complete. Charts saved in {os.path.join(src_dir, 'pics')}/")
