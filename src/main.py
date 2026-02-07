import os
import sys

# 1. Dynamic Imports
# Since main.py is inside 'src', we import neighbors directly.
try:
    from loader import load_and_preprocess
    from plots import plot_all_eda, plot_final_forecast
    from model import check_stationarity, train_and_forecast
except ImportError as e:
    # Fallback in case python doesn't see the current folder
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from loader import load_and_preprocess
    from plots import plot_all_eda, plot_final_forecast
    from model import check_stationarity, train_and_forecast

if __name__ == "__main__":
    print("=== AVOCADO SALES FORECASTING (Running from src) ===\n")

    # 2. Dynamic Path Finding
    # Get the folder where THIS file (main.py) is located
    current_src_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the project root
    project_root = os.path.dirname(current_src_folder)
    
    # Construct the path to the data
    data_path = os.path.join(project_root, 'data', 'avocado.csv')

    print(f"--> Data Path Detected: {data_path}")

    if not os.path.exists(data_path):
        print(f"ERROR: File not found at {data_path}")
        print("Please check that 'avocado.csv' is in the 'data' folder next to 'src'.")
        sys.exit(1)

    # 3. Execution Pipeline
    # Load
    ts = load_and_preprocess(data_path)

    # EDA (Will save images to the src folder)
    plot_all_eda(ts)

    # Stats
    check_stationarity(ts)

    # Train
    train, test, forecast, conf_int, mape = train_and_forecast(ts)

    # Forecast Plot
    plot_final_forecast(train, test, forecast, conf_int, mape)

    print(f"\n=== SUCCESS: Analysis complete. Images saved in {current_src_folder} ===")