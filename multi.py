import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time as t
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json
import os

# Configure Streamlit
st.set_page_config(
    page_title="Advanced BTC Trading Bars",
    page_icon="₿",
    layout="wide"
)

# Streamlit sidebar parameters
st.sidebar.title("Trading Parameters")

# Symbol selection
popular_cryptos = {
    'Bitcoin (BTC)': 'BTCUSDT',
    'Ethereum (ETH)': 'ETHUSDT', 
    'Solana (SOL)': 'SOLUSDT',
    'Cardano (ADA)': 'ADAUSDT',
    'XRP': 'XRPUSDT',
    'Dogecoin (DOGE)': 'DOGEUSDT',
    'Avalanche (AVAX)': 'AVAXUSDT',
    'Polygon (MATIC)': 'MATICUSDT',
    'Chainlink (LINK)': 'LINKUSDT',
    'Uniswap (UNI)': 'UNIUSDT',
    'Litecoin (LTC)': 'LTCUSDT',
    'Bitcoin Cash (BCH)': 'BCHUSDT',
    'Algorand (ALGO)': 'ALGOUSDT',
    'VeChain (VET)': 'VETUSDT',
    'Cosmos (ATOM)': 'ATOMUSDT'
}

selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency", 
    options=list(popular_cryptos.keys()),
    index=0  # Default to Bitcoin
)

# Option for custom symbol
use_custom = st.sidebar.checkbox("Use custom symbol")
if use_custom:
    custom_symbol = st.sidebar.text_input("Custom Symbol (e.g., ADAUSDT)", value="ADAUSDT").upper()
    selected_symbol = custom_symbol
    base_asset = custom_symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
else:
    selected_symbol = popular_cryptos[selected_crypto]
    base_asset = selected_symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')

st.sidebar.markdown(f"**Trading Pair:** {selected_symbol}")

# Bar parameters
st.sidebar.subheader("Bar Parameters")
interval_sec = st.sidebar.number_input("Time Bar Interval (sec)", value=60, min_value=1)
tick_threshold = st.sidebar.number_input("Tick Bar Threshold (# trades)", value=100, min_value=1)

# Dynamic volume threshold based on asset
default_vol_thresholds = {
    'BTC': 10.0, 'ETH': 50.0, 'SOL': 1000.0, 'ADA': 10000.0, 'XRP': 10000.0,
    'DOGE': 100000.0, 'AVAX': 1000.0, 'MATIC': 10000.0, 'LINK': 1000.0,
    'UNI': 1000.0, 'LTC': 100.0, 'BCH': 50.0, 'ALGO': 10000.0, 'VET': 100000.0, 'ATOM': 1000.0
}

default_vol = default_vol_thresholds.get(base_asset, 1000.0)
vol_threshold = st.sidebar.number_input(f"Volume Bar Threshold ({base_asset})", value=default_vol, min_value=0.001, step=default_vol/10)

# Dynamic dollar threshold based on asset price ranges
default_dollar_thresholds = {
    'BTC': 200000.0, 'ETH': 100000.0, 'SOL': 50000.0, 'ADA': 10000.0, 'XRP': 10000.0,
    'DOGE': 5000.0, 'AVAX': 25000.0, 'MATIC': 5000.0, 'LINK': 25000.0,
    'UNI': 25000.0, 'LTC': 50000.0, 'BCH': 50000.0, 'ALGO': 5000.0, 'VET': 1000.0, 'ATOM': 25000.0
}

default_dollar = default_dollar_thresholds.get(base_asset, 25000.0)
doll_threshold = st.sidebar.number_input("Dollar Bar Threshold (USD)", value=default_dollar, min_value=100.0, step=1000.0)

lookback_days = st.sidebar.number_input("Days of Historical Data", value=1, min_value=1, max_value=30)
max_trades = st.sidebar.number_input("Max Trades to Fetch (0 for no limit)", value=100000, min_value=0, step=10000)

# Advanced options
st.sidebar.subheader("Advanced Options")
use_cache = st.sidebar.checkbox("Use cached data", value=True)
batch_size = st.sidebar.selectbox("Batch size", [500, 1000], index=1)
delay_ms = st.sidebar.slider("Delay between requests (ms)", 100, 2000, 500, 50)

class ImprovedBinanceFetcher:
    def __init__(self, delay_ms=500, batch_size=1000):
        self.base_url = 'https://api.binance.com/api/v3/aggTrades'
        self.delay = delay_ms / 1000.0
        self.batch_size = batch_size
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_batch(self, params, semaphore):
        """Fetch a single batch with rate limiting"""
        async with semaphore:
            try:
                await asyncio.sleep(self.delay)  # Rate limiting
                async with self.session.get(self.base_url, params=params) as response:
                    if response.status == 429:  # Rate limited
                        st.warning("Rate limited, increasing delay...")
                        await asyncio.sleep(5)
                        return None
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                st.error(f"Error in batch fetch: {e}")
                return None
    
    async def fetch_trades_async(self, symbol, days, max_records):
        """Async version of trade fetching with better rate limiting"""
        end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
        start_ts = end_ts - days * 24 * 3600 * 1000
        
        all_trades = []
        current_ts = start_ts
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_count = 0
        
        while current_ts < end_ts and (not max_records or len(all_trades) < max_records):
            params = {
                'symbol': symbol,
                'startTime': current_ts,
                'endTime': min(current_ts + 3600000, end_ts),  # 1 hour chunks
                'limit': self.batch_size
            }
            
            batch_count += 1
            status_text.text(f"Fetching batch {batch_count}, records: {len(all_trades)}")
            
            data = await self.fetch_batch(params, semaphore)
            
            if not data:
                break
                
            if isinstance(data, dict) and 'code' in data:
                st.error(f"API Error: {data}")
                break
                
            all_trades.extend(data)
            
            if len(data) < self.batch_size:
                break
                
            current_ts = data[-1]['T'] + 1
            
            # Update progress
            progress = min((current_ts - start_ts) / (end_ts - start_ts), 1.0)
            progress_bar.progress(progress)
            
            if max_records and len(all_trades) >= max_records:
                all_trades = all_trades[:max_records]
                break
        
        progress_bar.empty()
        status_text.empty()
        return all_trades

def fetch_trades_sync_chunked(symbol: str, days: int, max_records: int, delay_ms: int = 500, batch_size: int = 1000) -> pd.DataFrame:
    """
    Improved synchronous fetching with better chunking and rate limiting.
    """
    base_url = 'https://api.binance.com/api/v3/aggTrades'
    end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ts = end_ts - days * 24 * 3600 * 1000
    
    all_trades = []
    current_ts = start_ts
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    consecutive_errors = 0
    max_errors = 5
    
    batch_count = 0
    
    while current_ts < end_ts and (not max_records or len(all_trades) < max_records):
        # Use smaller time chunks to avoid rate limits
        chunk_end = min(current_ts + 1800000, end_ts)  # 30 min chunks
        
        params = {
            'symbol': symbol,
            'startTime': current_ts,
            'endTime': chunk_end,
            'limit': batch_size
        }
        
        batch_count += 1
        status_text.text(f"Fetching batch {batch_count}, records: {len(all_trades)}")
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 429:  # Rate limited
                st.warning("Rate limited. Increasing delay...")
                t.sleep(5)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if not data or (isinstance(data, dict) and 'code' in data):
                st.warning(f"No data or API error for batch {batch_count}")
                break
            
            all_trades.extend(data)
            consecutive_errors = 0
            
            # Update timestamp
            if len(data) < batch_size:
                current_ts = chunk_end + 1
            else:
                current_ts = data[-1]['T'] + 1
            
            # Update progress
            progress = min((current_ts - start_ts) / (end_ts - start_ts), 1.0)
            progress_bar.progress(progress)
            
            # Rate limiting delay
            t.sleep(delay_ms / 1000.0)
            
            if max_records and len(all_trades) >= max_records:
                all_trades = all_trades[:max_records]
                break
                
        except requests.exceptions.RequestException as e:
            consecutive_errors += 1
            st.warning(f"Request error (attempt {consecutive_errors}): {e}")
            
            if consecutive_errors >= max_errors:
                st.error("Too many consecutive errors. Stopping fetch.")
                break
                
            t.sleep(2 ** consecutive_errors)  # Exponential backoff
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_trades:
        st.error("No trades fetched.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    df['time'] = pd.to_datetime(df['T'], unit='ms')
    df['price'] = df['p'].astype(float)
    df['qty'] = df['q'].astype(float)
    df['side'] = np.where(df['m'], -1, 1)
    df['dollar'] = df['price'] * df['qty']
    df = df[['time', 'price', 'qty', 'side', 'dollar']].sort_values('time')
    
    return df

# Cache trades data
@st.cache_data(ttl=3600)
def cached_fetch_trades(symbol: str, days: int, max_records: int, delay_ms: int, batch_size: int):
    return fetch_trades_sync_chunked(symbol, days, max_records, delay_ms, batch_size)

# Alternative: Use async fetching (experimental)
async def async_fetch_wrapper(symbol, days, max_records, delay_ms, batch_size):
    async with ImprovedBinanceFetcher(delay_ms, batch_size) as fetcher:
        return await fetcher.fetch_trades_async(symbol, days, max_records)

# Fetch trades
st.header(f"📈 {selected_crypto} Trading Bars Analysis")

col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"Enhanced analysis for **{selected_symbol}** with better rate limiting and larger data fetching capabilities")
with col2:
    use_async = st.checkbox("Use async fetching (experimental)", value=False)

with st.spinner(f"Fetching {selected_symbol} trades from Binance..."):
    if use_async:
        # Async version (requires more setup but potentially faster)
        st.info("Using experimental async fetching...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            trades_data = loop.run_until_complete(
                async_fetch_wrapper(selected_symbol, lookback_days, max_trades if max_trades > 0 else 0, delay_ms, batch_size)
            )
            trades = pd.DataFrame(trades_data)
            if not trades.empty:
                trades['time'] = pd.to_datetime(trades['T'], unit='ms')
                trades['price'] = trades['p'].astype(float)
                trades['qty'] = trades['q'].astype(float)
                trades['side'] = np.where(trades['m'], -1, 1)
                trades['dollar'] = trades['price'] * trades['qty']
                trades = trades[['time', 'price', 'qty', 'side', 'dollar']].sort_values('time')
        except Exception as e:
            st.error(f"Async fetch failed: {e}. Falling back to sync method.")
            trades = cached_fetch_trades(selected_symbol, lookback_days, max_trades if max_trades > 0 else 0, delay_ms, batch_size) if use_cache else fetch_trades_sync_chunked(selected_symbol, lookback_days, max_trades if max_trades > 0 else 0, delay_ms, batch_size)
    else:
        # Sync version with caching
        trades = cached_fetch_trades(selected_symbol, lookback_days, max_trades if max_trades > 0 else 0, delay_ms, batch_size) if use_cache else fetch_trades_sync_chunked(selected_symbol, lookback_days, max_trades if max_trades > 0 else 0, delay_ms, batch_size)

# Display fetched data info
if not trades.empty:
    start_time = trades['time'].min()
    end_time = trades['time'].max()
    duration = end_time - start_time
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", f"{len(trades):,}")
    with col2:
        st.metric("Time Span", f"{duration.total_seconds()/3600:.1f}h")
    with col3:
        st.metric("Avg Trade Size", f"{trades['qty'].mean():.4f} {base_asset}")
    with col4:
        st.metric("Total Volume", f"{trades['qty'].sum():.1f} {base_asset}")
    
    st.success(f"Successfully fetched {len(trades):,} {selected_symbol} trades from {start_time} to {end_time}")
else:
    st.error("No trades were fetched.")
    st.stop()

# Bar-building functions (keeping your original logic)
def get_time_bars(trades, interval_sec):
    df = trades.sort_values('time').copy()
    df['bin'] = ((df['time'] - df['time'].iloc[0]).dt.total_seconds() / interval_sec).astype(int)
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
        duration=('time', lambda x: (x.max() - x.min()).total_seconds())
    )
    bars['volatility'] = bars['high'] - bars['low']
    return bars

def get_tick_bars(trades, tick_threshold):
    df = trades.sort_values('time').copy()
    df['bin'] = (np.arange(len(df)) // tick_threshold).astype(int)
    grouped = df.groupby('bin', as_index=False)
    bars = grouped.agg(
        time=('time', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum'),
        dollar=('dollar', 'sum'),
        n_trades=('price', 'count'),
        duration=('time', lambda x: (x.max() - x.min()).total_seconds())
    )
    bars['volatility'] = bars['high'] - bars['low']
    return bars

def get_volume_bars(trades, vol_threshold):
    df = trades.sort_values('time').copy()
    df['cum_vol'] = df['qty'].cumsum()
    df['bin'] = (df['cum_vol'] / vol_threshold).astype(int)
    grouped = df.groupby('bin', as_index=False)
    bars = grouped.agg(
        time=('time', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum'),
        dollar=('dollar', 'sum'),
        n_trades=('price', 'count'),
        duration=('time', lambda x: (x.max() - x.min()).total_seconds())
    )
    bars['volatility'] = bars['high'] - bars['low']
    return bars

def get_dollar_bars(trades, doll_threshold):
    df = trades.sort_values('time').copy()
    df['cum_doll'] = df['dollar'].cumsum()
    df['bin'] = (df['cum_doll'] / doll_threshold).astype(int)
    grouped = df.groupby('bin', as_index=False)
    bars = grouped.agg(
        time=('time', 'last'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum'),
        dollar=('dollar', 'sum'),
        n_trades=('price', 'count'),
        duration=('time', lambda x: (x.max() - x.min()).total_seconds())
    )
    bars['volatility'] = bars['high'] - bars['low']
    return bars

# Build bar series
with st.spinner("Building different bar types..."):
    bars_dict = {
        'Time': get_time_bars(trades, interval_sec),
        'Tick': get_tick_bars(trades, tick_threshold),
        'Volume': get_volume_bars(trades, vol_threshold),
        'Dollar': get_dollar_bars(trades, doll_threshold)
    }

# Display bar statistics
st.header("Bar Statistics")
stats_data = []
for name, bars in bars_dict.items():
    stats_data.append({
        'Bar Type': name,
        'Number of Bars': len(bars),
        'Avg Duration (min)': bars['duration'].mean() / 60 if len(bars) > 0 else 0,
        'Avg Volume per Bar': bars['volume'].mean() if len(bars) > 0 else 0,
        'Avg Volatility': bars['volatility'].mean() if len(bars) > 0 else 0
    })

st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# Display charts
st.header("Candlestick Charts")
for name, bars in bars_dict.items():
    with st.expander(f"{name} Bars Chart", expanded=True):
        if len(bars) < 2:
            st.warning(f"Not enough data to plot {name} bars (only {len(bars)} bars).")
            continue
        
        df_plot = bars.set_index('time')[['open', 'high', 'low', 'close', 'volume']]
        
        try:
            fig, ax = mpf.plot(
                df_plot,
                type='candle', 
                volume=True, 
                returnfig=True,
                title=f"{selected_symbol} {name} Bars ({len(bars)} bars)",
                style='yahoo',
                figsize=(12, 8)
            )
            st.pyplot(fig)
            
            # Show recent bars
            with st.expander(f"Recent {name} Bars Data"):
                st.dataframe(bars.tail(10).round(4))
                
        except Exception as e:
            st.error(f"Error plotting {name} bars: {e}")

# Correlation & predictive model
st.header("Correlation & Predictive Accuracy")

def run_model(bars):
    if len(bars) < 20:  # Need enough data for meaningful model
        return np.nan
    
    df = bars.copy()
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Add more features
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['return'].shift(lag)
    
    df['vol_lag_1'] = df['volatility'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)
    
    df = df.dropna()
    
    if len(df) < 20:
        return np.nan
    
    feature_cols = [f'lag_{lag}' for lag in range(1, 6)] + ['vol_lag_1', 'volume_lag_1']
    X = df[feature_cols]
    y = (df['return'] > 0).astype(int)
    
    if len(X) < 20:
        return np.nan
    
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        model.fit(Xtr, ytr)
        return model.score(Xte, yte)
    except Exception as e:
        st.warning(f"Model fitting error: {e}")
        return np.nan

# Calculate correlations and model accuracy
results = {}
for name, bars in bars_dict.items():
    if len(bars) > 1:
        # Correlation between duration and volatility (for time bars) or n_trades and volatility (for others)
        if name == 'Time':
            corr = bars['duration'].corr(bars['volatility'])
        else:
            corr = bars['n_trades'].corr(bars['volatility'])
        
        accuracy = run_model(bars)
        
        results[name] = {
            'Correlation (Duration/Trades vs Volatility)': corr,
            'Predictive Accuracy': accuracy,
            'Number of Bars': len(bars)
        }
    else:
        results[name] = {
            'Correlation (Duration/Trades vs Volatility)': np.nan,
            'Predictive Accuracy': np.nan,
            'Number of Bars': len(bars)
        }

results_df = pd.DataFrame(results).T
st.dataframe(results_df.round(4), use_container_width=True)

# Performance tips
st.sidebar.markdown("---")
st.sidebar.markdown("### Asset Information")
st.sidebar.info(f"""
**Current Selection:** {selected_symbol}
**Base Asset:** {base_asset}
**Volume Threshold:** {vol_threshold:,.1f} {base_asset}
**Dollar Threshold:** ${doll_threshold:,.0f}

Default thresholds are optimized for each asset's typical trading volumes and price ranges.
""")

st.sidebar.markdown("### Performance Tips")
st.sidebar.markdown("""
**To fetch more data:**
- Increase delay between requests (500ms+)
- Use smaller batch sizes (500 instead of 1000)
- Enable caching to avoid refetching
- Try async fetching for potentially better performance
- Consider fetching data in multiple sessions

**Rate Limit Info:**
- Binance: 1200 requests/minute
- Weight per request: 1
- Use delays of 50-100ms minimum
""")