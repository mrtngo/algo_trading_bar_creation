# # %%
# # financial_bars_streamlit.py
# # Streamlit app to fetch BTC/USDT data (via Binance public API), build and compare Time, Tick, Volume, Dollar bars

# # %%
# # 1. Imports
# import pandas as pd
# import numpy as np
# import requests
# import mplfinance as mpf
# import streamlit as st
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import time as t

# # %%
# # 2. Streamlit sidebar parameters
# st.sidebar.title("Bar Parameters")
# interval_sec = st.sidebar.number_input("Time Bar Interval (sec)", value=60, min_value=1)
# tick_threshold = st.sidebar.number_input("Tick Bar Threshold (# trades)", value=100, min_value=1)
# vol_threshold = st.sidebar.number_input("Volume Bar Threshold (BTC)", value=10.0, min_value=0.001, step=0.5)
# doll_threshold = st.sidebar.number_input("Dollar Bar Threshold (USD)", value=200000.0, min_value=100.0, step=1000.0)
# lookback_days = st.sidebar.number_input("Days of Historical Data", value=1, min_value=1, max_value=30)
# max_trades = st.sidebar.number_input("Max Trades to Fetch", value=50000, min_value=1000, step=1000)

# # %%
# # %%
# # 3. Fetch BTC/USDT aggTrades with feedback
# @st.cache_data(ttl=3600)
# def fetch_trades(symbol: str, days: int, max_records: int) -> pd.DataFrame:
#     """
#     Fetch aggregated trades from Binance public API.
#     Args:
#         symbol: Trading pair, e.g. "BTCUSDT".
#         days: Lookback period in days.
#         max_records: Maximum number of trades to fetch.
#     Returns:
#         DataFrame with columns [time, price, qty, side, dollar].
#     """
#     base_url = 'https://api.binance.com/api/v3/aggTrades'
#     end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
#     start_ts = end_ts - days * 24 * 3600 * 1000
#     all_trades = []
#     params = {'symbol': symbol, 'startTime': start_ts, 'endTime': end_ts, 'limit': 1000}
#     pbar = st.progress(0)
#     records = 0
#     while True:
#         resp = requests.get(base_url, params=params)
#         data = resp.json()
#         if not data or records >= max_records:
#             break
#         all_trades.extend(data)
#         records += len(data)
#         pbar.progress(min(records / max_records, 1.0))
#         last_ts = data[-1]['T']
#         if len(data) < 1000 or last_ts >= end_ts:
#             break
#         params['startTime'] = last_ts + 1
#         t.sleep(0.2)
#     pbar.empty()
#     df = pd.DataFrame(all_trades[:max_records])
#     df['time'] = df['T'] / 1000.0
#     df['price'] = df['p'].astype(float)
#     df['qty'] = df['q'].astype(float)
#     df['side'] = np.where(df['m'], -1, 1)
#     df['dollar'] = df['price'] * df['qty']
#     return df[['time', 'price', 'qty', 'side', 'dollar']]

# # Call fetch_trades with current slider values
# with st.spinner("Fetching trades..."):
#     trades = fetch_trades("BTCUSDT", lookback_days, max_trades)

# # %%
# # 4. Bar-building functions
# def get_time_bars(trades, interval_sec):
#     df = trades.copy()
#     df['bin'] = np.floor((df['time'] - df['time'].iloc[0]) / interval_sec)
#     grouped = df.groupby('bin', as_index=False)
#     bars = grouped.agg(
#         time=('time', 'first'),
#         open=('price', 'first'),
#         high=('price', 'max'),
#         low=('price', 'min'),
#         close=('price', 'last'),
#         volume=('qty', 'sum'),
#         dollar=('dollar', 'sum'),
#         n_trades=('price', 'count'),
#         duration=('time', lambda x: x.max() - x.min())
#     )
#     bars['volatility'] = bars['high'] - bars['low']
#     return bars


# def get_tick_bars(trades, tick_threshold):
#     bars = []
#     start = 0
#     for i in range(len(trades)):
#         if (i - start + 1) >= tick_threshold:
#             seg = trades.iloc[start:i+1]
#             bars.append({
#                 'time': seg['time'].iloc[-1],
#                 'open': seg['price'].iloc[0],
#                 'high': seg['price'].max(),
#                 'low': seg['price'].min(),
#                 'close': seg['price'].iloc[-1],
#                 'volume': seg['qty'].sum(),
#                 'dollar': seg['dollar'].sum(),
#                 'n_trades': len(seg),
#                 'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
#             })
#             start = i + 1
#     df = pd.DataFrame(bars)
#     df['volatility'] = df['high'] - df['low']
#     return df


# def get_volume_bars(trades, vol_threshold):
#     bars = []
#     cum_vol = 0
#     start = 0
#     for i, row in trades.iterrows():
#         cum_vol += row['qty']
#         if cum_vol >= vol_threshold:
#             seg = trades.iloc[start:i+1]
#             bars.append({
#                 'time': seg['time'].iloc[-1],
#                 'open': seg['price'].iloc[0],
#                 'high': seg['price'].max(),
#                 'low': seg['price'].min(),
#                 'close': seg['price'].iloc[-1],
#                 'volume': seg['qty'].sum(),
#                 'dollar': seg['dollar'].sum(),
#                 'n_trades': len(seg),
#                 'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
#             })
#             start = i + 1
#             cum_vol = 0
#     df = pd.DataFrame(bars)
#     df['volatility'] = df['high'] - df['low']
#     return df


# def get_dollar_bars(trades, doll_threshold):
#     bars = []
#     cum_doll = 0
#     start = 0
#     for i, row in trades.iterrows():
#         cum_doll += row['dollar']
#         if cum_doll >= doll_threshold:
#             seg = trades.iloc[start:i+1]
#             bars.append({
#                 'time': seg['time'].iloc[-1],
#                 'open': seg['price'].iloc[0],
#                 'high': seg['price'].max(),
#                 'low': seg['price'].min(),
#                 'close': seg['price'].iloc[-1],
#                 'volume': seg['qty'].sum(),
#                 'dollar': seg['dollar'].sum(),
#                 'n_trades': len(seg),
#                 'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
#             })
#             start = i + 1
#             cum_doll = 0
#     df = pd.DataFrame(bars)
#     df['volatility'] = df['high'] - df['low']
#     return df

# # %%
# # 5. Build bar series
# bars_dict = {
#     'Time': get_time_bars(trades, interval_sec),
#     'Tick': get_tick_bars(trades, tick_threshold),
#     'Volume': get_volume_bars(trades, vol_threshold),
#     'Dollar': get_dollar_bars(trades, doll_threshold)
# }

# # %%
# # 6. Display charts
# st.header("Candlestick Charts")
# for name, bars in bars_dict.items():
#     st.subheader(f"{name} Bars")
#     df_plot = bars.set_index(pd.to_datetime(bars['time'], unit='s'))
#     fig, ax = mpf.plot(
#         df_plot[['open', 'high', 'low', 'close', 'volume']],
#         type='candle', volume=True, returnfig=True
#     )
#     st.pyplot(fig)

# # %%
# # 7. Correlation & predictive model
# st.header("Correlation & Predictive Accuracy")
# corrs = {
#     name: (bars['n_trades'] if name == 'Time' else bars['duration']).corr(bars['volatility'])
#     for name, bars in bars_dict.items()
# }

# def run_model(bars):
#     df = bars.copy()
#     df['return'] = np.log(df['close'] / df['close'].shift(1))
#     for lag in range(1, 6):
#         df[f'lag_{lag}'] = df['return'].shift(lag)
#     df = df.dropna()
#     X = df[[f'lag_{lag}' for lag in range(1, 6)]]
#     y = (df['return'] > 0).astype(int)
#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
#     model = LogisticRegression(solver='liblinear')
#     model.fit(Xtr, ytr)
#     return model.score(Xte, yte)

# results = {
#     name: {'correlation': corrs[name], 'accuracy': run_model(bars)}
#     for name, bars in bars_dict.items()
# }

# st.table(pd.DataFrame(results).T)


# financial_bars_streamlit.py
# Streamlit app to fetch BTC/USDT data (via Binance public API), build and compare Time, Tick, Volume, Dollar bars



# 1. Imports
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time as t
import json

# 2. Streamlit sidebar parameters
st.title("Financial Bars Analysis")
st.sidebar.title("Bar Parameters")
interval_sec = st.sidebar.number_input("Time Bar Interval (sec)", value=60, min_value=1)
tick_threshold = st.sidebar.number_input("Tick Bar Threshold (# trades)", value=100, min_value=1)
vol_threshold = st.sidebar.number_input("Volume Bar Threshold (BTC)", value=10.0, min_value=0.001, step=0.5)
doll_threshold = st.sidebar.number_input("Dollar Bar Threshold (USD)", value=200000.0, min_value=100.0, step=1000.0)
lookback_days = st.sidebar.number_input("Days of Historical Data", value=1, min_value=1, max_value=30)
max_trades = st.sidebar.number_input("Max Trades to Fetch", value=50000, min_value=1000, step=1000)

# 3. Fetch BTC-PERP trades from Binance public API (corrected)
@st.cache_data(ttl=3600)
def fetch_trades_binance(symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch recent trades from Binance public API.
    Args:
        symbol: Trading pair, e.g. "BTCUSDT".
        limit: Number of trades to fetch (max 1000 per request).
    Returns:
        DataFrame with columns [time, price, qty, side, dollar].
    """
    base_url = 'https://api.binance.com/api/v3/aggTrades'
    
    try:
        # Get recent trades
        params = {
            'symbol': symbol,
            'limit': min(limit, 1000)  # Binance API limit
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code != 200:
            st.error(f"Error fetching trades from Binance: {response.status_code}")
            return pd.DataFrame(columns=['time', 'price', 'qty', 'side', 'dollar'])
        
        data = response.json()
        
        if not data:
            st.warning("No trade data received from Binance")
            return pd.DataFrame(columns=['time', 'price', 'qty', 'side', 'dollar'])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Rename and convert columns
        df = df.rename(columns={
            'T': 'time',
            'p': 'price', 
            'q': 'qty',
            'm': 'side'
        })
        
        # Convert data types
        df['time'] = df['time'] / 1000.0  # ms to seconds
        df['price'] = df['price'].astype(float)
        df['qty'] = df['qty'].astype(float)
        
        # Convert side: True means buyer is market maker (sell), False means buyer is taker (buy)
        df['side'] = np.where(df['side'] == True, -1, 1)
        
        # Calculate dollar volume
        df['dollar'] = df['price'] * df['qty']
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        return df[['time', 'price', 'qty', 'side', 'dollar']]
        
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
        return pd.DataFrame(columns=['time', 'price', 'qty', 'side', 'dollar'])
    except Exception as e:
        st.error(f"Error processing trade data: {e}")
        return pd.DataFrame(columns=['time', 'price', 'qty', 'side', 'dollar'])

# Fetch trades with spinner
with st.spinner("Fetching trades from Binance..."):
    trades = fetch_trades_binance("BTCUSDT", max_trades)

# 4. Bar-building functions
def get_time_bars(trades, interval_sec):
    if trades.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility'])
    
    df = trades.copy()
    df['bin'] = np.floor((df['time'] - df['time'].iloc[0]) / interval_sec)
    grouped = df.groupby('bin', as_index=False)
    
    bars = grouped.agg(
        time=('time', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum'),
        dollar=('dollar', 'sum'),
        n_trades=('price', 'count'),
        duration=('time', lambda x: x.max() - x.min())
    )
    bars['volatility'] = bars['high'] - bars['low']
    return bars

def get_tick_bars(trades, tick_threshold):
    if trades.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility'])
    
    bars = []
    start = 0
    
    while start < len(trades):
        end = min(start + tick_threshold, len(trades))
        seg = trades.iloc[start:end]
        
        if len(seg) == 0:
            break
            
        bars.append({
            'time': seg['time'].iloc[-1],
            'open': seg['price'].iloc[0],
            'high': seg['price'].max(),
            'low': seg['price'].min(),
            'close': seg['price'].iloc[-1],
            'volume': seg['qty'].sum(),
            'dollar': seg['dollar'].sum(),
            'n_trades': len(seg),
            'duration': seg['time'].iloc[-1] - seg['time'].iloc[0] if len(seg) > 1 else 0
        })
        start = end
        
    df = pd.DataFrame(bars)
    if not df.empty:
        df['volatility'] = df['high'] - df['low']
    return df

def get_volume_bars(trades, vol_threshold):
    if trades.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility'])
    
    bars = []
    cum_vol = 0
    start = 0
    
    for i, row in trades.iterrows():
        cum_vol += row['qty']
        if cum_vol >= vol_threshold:
            seg = trades.iloc[start:i+1]
            bars.append({
                'time': seg['time'].iloc[-1],
                'open': seg['price'].iloc[0],
                'high': seg['price'].max(),
                'low': seg['price'].min(),
                'close': seg['price'].iloc[-1],
                'volume': seg['qty'].sum(),
                'dollar': seg['dollar'].sum(),
                'n_trades': len(seg),
                'duration': seg['time'].iloc[-1] - seg['time'].iloc[0] if len(seg) > 1 else 0
            })
            start = i + 1
            cum_vol = 0
            
    df = pd.DataFrame(bars)
    if not df.empty:
        df['volatility'] = df['high'] - df['low']
    return df

def get_dollar_bars(trades, doll_threshold):
    if trades.empty:
        return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility'])
    
    bars = []
    cum_doll = 0
    start = 0
    
    for i, row in trades.iterrows():
        cum_doll += row['dollar']
        if cum_doll >= doll_threshold:
            seg = trades.iloc[start:i+1]
            bars.append({
                'time': seg['time'].iloc[-1],
                'open': seg['price'].iloc[0],
                'high': seg['price'].max(),
                'low': seg['price'].min(),
                'close': seg['price'].iloc[-1],
                'volume': seg['qty'].sum(),
                'dollar': seg['dollar'].sum(),
                'n_trades': len(seg),
                'duration': seg['time'].iloc[-1] - seg['time'].iloc[0] if len(seg) > 1 else 0
            })
            start = i + 1
            cum_doll = 0
            
    df = pd.DataFrame(bars)
    if not df.empty:
        df['volatility'] = df['high'] - df['low']
    return df

# 5. Build bar series
if trades.empty:
    st.warning("No trades fetched. Please check your internet connection or try again later.")
    st.stop()

st.success(f"Fetched {len(trades)} trades successfully!")

bars_dict = {
    'Time': get_time_bars(trades, interval_sec),
    'Tick': get_tick_bars(trades, tick_threshold),
    'Volume': get_volume_bars(trades, vol_threshold),
    'Dollar': get_dollar_bars(trades, doll_threshold)
}

# 6. Display charts
st.header("Candlestick Charts")

for name, bars in bars_dict.items():
    if bars.empty:
        st.warning(f"No {name} bars generated with current parameters.")
        continue
        
    st.subheader(f"{name} Bars (Count: {len(bars)})")
    
    try:
        # Create datetime index
        df_plot = bars.copy()
        df_plot.index = pd.to_datetime(df_plot['time'], unit='s')
        
        # Ensure we have the required columns and they're numeric
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df_plot.columns:
                st.error(f"Missing column: {col}")
                continue
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
        
        # Remove any rows with NaN values
        df_plot = df_plot.dropna(subset=required_cols)
        
        if len(df_plot) < 2:
            st.warning(f"Not enough valid data points for {name} bars chart.")
            continue
            
        # Create the plot
        fig, axes = mpf.plot(
            df_plot[required_cols],
            type='candle',
            volume=True,
            style='charles',
            title=f'{name} Bars',
            returnfig=True,
            figsize=(12, 8)
        )
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating {name} bars chart: {e}")

# 7. Display bar statistics
st.header("Bar Statistics")
stats_data = []
for name, bars in bars_dict.items():
    if not bars.empty:
        stats_data.append({
            'Bar Type': name,
            'Count': len(bars),
            'Avg Duration (s)': bars['duration'].mean(),
            'Avg Volume': bars['volume'].mean(),
            'Avg Dollar Volume': bars['dollar'].mean(),
            'Avg Volatility': bars['volatility'].mean()
        })

if stats_data:
    st.dataframe(pd.DataFrame(stats_data))

# 8. Correlation & predictive model
st.header("Correlation & Predictive Accuracy")

def calculate_correlation(bars, name):
    if bars.empty or len(bars) < 2:
        return np.nan
    
    if name == 'Time':
        x = bars['n_trades']
    else:
        x = bars['duration']
    
    y = bars['volatility']
    
    # Remove any infinite or NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    
    return np.corrcoef(x[mask], y[mask])[0, 1]

def run_model(bars):
    if bars.empty or len(bars) < 10:  # Need minimum data for train/test split
        return np.nan
    
    try:
        df = bars.copy()
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Create lagged features
        for lag in range(1, 6):
            df[f'lag_{lag}'] = df['return'].shift(lag)
        
        df = df.dropna()
        
        if len(df) < 10:
            return np.nan
        
        X = df[[f'lag_{lag}' for lag in range(1, 6)]]
        y = (df['return'] > 0).astype(int)
        
        # Ensure we have enough data for train/test split
        if len(X) < 6:
            return np.nan
        
        test_size = max(0.2, 2/len(X))  # At least 2 samples for test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model = LogisticRegression(solver='liblinear', random_state=42)
        model.fit(X_train, y_train)
        
        return model.score(X_test, y_test)
        
    except Exception as e:
        st.warning(f"Error in predictive model: {e}")
        return np.nan

# Calculate results
results = {}
for name, bars in bars_dict.items():
    correlation = calculate_correlation(bars, name)
    accuracy = run_model(bars)
    results[name] = {
        'correlation': correlation,
        'accuracy': accuracy
    }

# Display results
results_df = pd.DataFrame(results).T
results_df.columns = ['Correlation', 'Predictive Accuracy']
st.dataframe(results_df)

# Add explanations
st.subheader("Explanation")
st.write("""
- **Correlation**: Measures the linear relationship between bar characteristics and volatility
- **Predictive Accuracy**: Performance of a logistic regression model predicting price direction using lagged returns
- **Time Bars**: Fixed time intervals
- **Tick Bars**: Fixed number of trades
- **Volume Bars**: Fixed volume thresholds
- **Dollar Bars**: Fixed dollar volume thresholds
""")