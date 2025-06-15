# financial_bars_streamlit.py
# Streamlit app to fetch BTC-PERP trade data (via Hyperliquid Python SDK), build and compare Time, Tick, Volume, Dollar bars

# Imports
import pandas as pd
import numpy as np
import streamlit as st
import mplfinance as mpf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time as t
import json
from hyperliquid import Info
from hyperliquid.utils import constants

# Streamlit sidebar parameters
st.sidebar.title("Bar Parameters")
interval_sec = st.sidebar.number_input("Time Bar Interval (sec)", value=60, min_value=1)
tick_threshold = st.sidebar.number_input("Tick Bar Threshold (# trades)", value=100, min_value=1)
vol_threshold = st.sidebar.number_input("Volume Bar Threshold (BTC)", value=10.0, min_value=0.001, step=0.5)
doll_threshold = st.sidebar.number_input("Dollar Bar Threshold (USD)", value=200000.0, min_value=100.0, step=1000.0)
lookback_days = st.sidebar.number_input("Days of Historical Data", value=1, min_value=1, max_value=30)
max_trades = st.sidebar.number_input("Max Trades to Fetch (0 for no limit)", value=500000, min_value=0, step=100000, help="Set to 0 to fetch all trades in the lookback period. Large values or 0 may take several minutes.")
if max_trades == 0 and lookback_days > 7:
    st.sidebar.warning("Fetching all trades for >7 days may take a long time.")

# Load Hyperliquid credentials
try:
    with open("examples/config.json", "r") as f:
        config = json.load(f)
    wallet_address = config["account_address"]
    private_key = config["secret_key"]
except FileNotFoundError:
    st.error("Please create 'examples/config.json' with 'account_address' and 'secret_key'.")
    st.stop()
except KeyError:
    st.error("'config.json' must contain 'account_address' and 'secret_key'.")
    st.stop()

# Fetch BTC-PERP trades using Hyperliquid Python SDK
@st.cache_data(ttl=3600)
def fetch_trades_hyperliquid(coin: str, days: int, max_records: int, _wallet_address: str, _private_key: str) -> pd.DataFrame:
    """
    Fetch historical user fills for BTC-PERP via Hyperliquid Python SDK.
    Args:
        coin: Coin name, e.g., "BTC" for BTC-PERP.
        days: Lookback period in days.
        max_records: Maximum number of trades to fetch (0 for no limit).
        _wallet_address: User wallet address (not hashed for cache).
        _private_key: Private key (not hashed for cache).
    Returns:
        DataFrame with columns [time, price, qty, side, dollar].
    """
    try:
        info = Info(base_url=constants.MAINNET_API_URL, skip_ws=True)
        end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ts = end_ts - days * 24 * 3600 * 1000
        # Fetch user fills (requires authentication via wallet address)
        raw_trades = info.user_fills_by_time(address=wallet_address, start_time=start_ts, end_time=end_ts)
        if not raw_trades:
            st.warning(f"No trades found for {coin} in the specified period. Ensure your account has BTC-PERP trade history.")
            return pd.DataFrame()
        df = pd.DataFrame(raw_trades)
        # Normalize columns to match expected format
        df = df.rename(columns={"px": "price", "sz": "qty"})
        df["time"] = df["time"] / 1000.0  # Convert ms to s
        df["price"] = df["price"].astype(float)
        df["qty"] = df["qty"].astype(float)
        df["side"] = df["side"].map({"A": -1, "B": 1})  # A=sell, B=buy
        df["dollar"] = df["price"] * df["qty"]
        df = df[["time", "price", "qty", "side", "dollar"]].sort_values("time")
        return df.head(max_records) if max_records else df
    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        return pd.DataFrame()

# Call fetch_trades_hyperliquid
with st.spinner("Fetching trades from Hyperliquid SDK..."):
    trades = fetch_trades_hyperliquid("BTC", lookback_days, max_trades, wallet_address, private_key)

# Display fetched data range
if not trades.empty:
    start_time = pd.to_datetime(trades["time"].min(), unit="s")
    end_time = pd.to_datetime(trades["time"].max(), unit="s")
    st.write(f"Fetched {len(trades)} trades from {start_time} to {end_time}")
else:
    st.warning("No trades fetched. Ensure your account has BTC-PERP trades or adjust parameters.")
    st.stop()

# Optimized bar-building functions
def get_time_bars(trades, interval_sec):
    df = trades.sort_values("time").copy()
    df["bin"] = ((df["time"] - df["time"].iloc[0]) // interval_sec).astype(int)
    grouped = df.groupby("bin", as_index=False)
    bars = grouped.agg(
        time=("time", "first"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        n_trades=("price", "count"),
        duration=("time", lambda x: x.max() - x.min())
    )
    bars["volatility"] = bars["high"] - bars["low"]
    return bars

def get_tick_bars(trades, tick_threshold):
    df = trades.sort_values("time").copy()
    df["bin"] = (np.arange(len(df)) // tick_threshold).astype(int)
    grouped = df.groupby("bin", as_index=False)
    bars = grouped.agg(
        time=("time", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        n_trades=("price", "count"),
        duration=("time", lambda x: x.max() - x.min())
    )
    bars["volatility"] = bars["high"] - bars["low"]
    return bars

def get_volume_bars(trades, vol_threshold):
    df = trades.sort_values("time").copy()
    df["cum_vol"] = df["qty"].cumsum()
    df["bin"] = (df["cum_vol"] / vol_threshold).astype(int)
    grouped = df.groupby("bin", as_index=False)
    bars = grouped.agg(
        time=("time", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        n_trades=("price", "count"),
        duration=("time", lambda x: x.max() - x.min())
    )
    bars["volatility"] = bars["high"] - bars["low"]
    return bars

def get_dollar_bars(trades, doll_threshold):
    df = trades.sort_values("time").copy()
    df["cum_doll"] = df["dollar"].cumsum()
    df["bin"] = (df["cum_doll"] / doll_threshold).astype(int)
    grouped = df.groupby("bin", as_index=False)
    bars = grouped.agg(
        time=("time", "last"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        dollar=("dollar", "sum"),
        n_trades=("price", "count"),
        duration=("time", lambda x: x.max() - x.min())
    )
    bars["volatility"] = bars["high"] - bars["low"]
    return bars

# Build bar series
bars_dict = {
    "Time": get_time_bars(trades, interval_sec),
    "Tick": get_tick_bars(trades, tick_threshold),
    "Volume": get_volume_bars(trades, vol_threshold),
    "Dollar": get_dollar_bars(trades, doll_threshold)
}

# Display charts
st.header("Candlestick Charts")
for name, bars in bars_dict.items():
    st.subheader(f"{name} Bars")
    if len(bars) < 2:
        st.warning(f"Not enough data to plot {name} bars.")
        continue
    df_plot = bars.set_index(pd.to_datetime(bars["time"], unit="s"))[["open", "high", "low", "close", "volume"]]
    try:
        fig, ax = mpf.plot(
            df_plot,
            type="candle",
            volume=True,
            returnfig=True,
            title=f"{name} Bars",
            style="yahoo"
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Error plotting {name} bars: {e}")

# Correlation & predictive model
st.header("Correlation & Predictive Accuracy")
def run_model(bars):
    if len(bars) < 10:
        return np.nan
    df = bars.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["return"].shift(lag)
    df = df.dropna()
    if len(df) < 10:
        return np.nan
    X = df[[f"lag_{lag}" for lag in range(1, 6)]]
    y = (df["return"] > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(solver="liblinear")
    model.fit(Xtr, ytr)
    return model.score(Xte, yte)

corrs = {
    name: (bars["n_trades"] if name == "Time" else bars["duration"]).corr(bars["volatility"])
    for name, bars in bars_dict.items()
}
results = {
    name: {"correlation": corrs[name], "accuracy": run_model(bars)}
    for name, bars in bars_dict.items()
}
st.table(pd.DataFrame(results).T)