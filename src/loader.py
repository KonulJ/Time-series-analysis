import pandas as pd


def load_and_preprocess(filepath: str) -> pd.Series:
    print(f"\n[1/4] Loading data from {filepath}...")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    mask = (df["region"] == "TotalUS") & (df["type"] == "conventional")
    df = df[mask].sort_values("Date")

    ts = df.set_index("Date")["AveragePrice"].resample("W-MON").mean()
    ts = ts.interpolate(method="linear")

    print(f"      {len(ts)} weeks of data loaded.")
    return ts
