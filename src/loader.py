import pandas as pd

def load_and_preprocess(filepath):
    """
    Loads avocado data, filters for 'TotalUS' conventional,
    resamples to weekly frequency, and handles missing values.
    """
    print(f"\n[1/4] Loading Data from {filepath}...")
    
    # 1. Load
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"CRITICAL: '{filepath}' not found. Please move avocado.csv to the data folder.")

    # 2. Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 3. Filter (Univariate Focus)
    # We focus on TotalUS + Conventional for a clean trend
    mask = (df['region'] == 'TotalUS') & (df['type'] == 'conventional')
    df = df[mask].sort_values('Date')
    
    # 4. Set Index & Resample
    # Resample to Weekly (Monday-Start) to ensure consistent timeline
    ts = df.set_index('Date')['AveragePrice'].resample('W-MON').mean()
    
    # 5. Interpolate (Fill gaps)
    ts = ts.interpolate(method='linear')
    
    print(f"      Data Prepared: {len(ts)} weeks of data.")
    return ts