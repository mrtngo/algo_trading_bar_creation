import streamlit as st
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Hyperliquid BTC Trading Bars",
    page_icon="₿",
    layout="wide"
)

class HyperliquidAPI:
    def __init__(self):
        self.base_url = "https://api.hyperliquid.xyz"
    
    def get_recent_trades(self, symbol: str = "BTC", limit: int = 1000) -> List[Dict]:
        """Fetch recent trades from Hyperliquid API"""
        try:
            # Using the info endpoint to get recent trades
            url = f"{self.base_url}/info"
            payload = {
                "type": "recentTrades",
                "coin": symbol
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            trades = response.json()
            if not trades:
                return []
            
            # Limit the number of trades
            return trades[:limit]
            
        except Exception as e:
            st.error(f"Error fetching trades: {str(e)}")
            return []

class BarBuilder:
    @staticmethod
    def create_tick_bars(trades_df: pd.DataFrame, tick_threshold: int) -> pd.DataFrame:
        """Create tick bars based on number of trades"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        bars = []
        current_bar = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0,
            'timestamp': None,
            'tick_count': 0
        }
        
        for _, trade in trades_df.iterrows():
            price = trade['price']
            volume = trade['sz']
            timestamp = trade['time']
            
            if current_bar['open'] is None:
                current_bar['open'] = price
                current_bar['timestamp'] = timestamp
            
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price
            current_bar['volume'] += volume
            current_bar['tick_count'] += 1
            
            if current_bar['tick_count'] >= tick_threshold:
                bars.append(current_bar.copy())
                current_bar = {
                    'open': None,
                    'high': -np.inf,
                    'low': np.inf,
                    'close': None,
                    'volume': 0,
                    'timestamp': None,
                    'tick_count': 0
                }
        
        if current_bar['tick_count'] > 0:
            bars.append(current_bar)
        
        if not bars:
            return pd.DataFrame()
        
        bars_df = pd.DataFrame(bars)
        bars_df['datetime'] = pd.to_datetime(bars_df['timestamp'], unit='ms')
        bars_df.set_index('datetime', inplace=True)
        
        return bars_df[['open', 'high', 'low', 'close', 'volume']]
    
    @staticmethod
    def create_time_bars(trades_df: pd.DataFrame, time_threshold_minutes: int) -> pd.DataFrame:
        """Create time bars based on time intervals"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        trades_df['datetime'] = pd.to_datetime(trades_df['time'], unit='ms')
        trades_df.set_index('datetime', inplace=True)
        
        # Resample to time intervals
        bars = trades_df.resample(f'{time_threshold_minutes}T').agg({
            'price': ['first', 'max', 'min', 'last'],
            'sz': 'sum'
        })
        
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        bars = bars.dropna()
        
        return bars
    
    @staticmethod
    def create_volume_bars(trades_df: pd.DataFrame, volume_threshold: float) -> pd.DataFrame:
        """Create volume bars based on cumulative volume"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        bars = []
        current_bar = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0,
            'timestamp': None
        }
        
        for _, trade in trades_df.iterrows():
            price = trade['price']
            volume = trade['sz']
            timestamp = trade['time']
            
            if current_bar['open'] is None:
                current_bar['open'] = price
                current_bar['timestamp'] = timestamp
            
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price
            current_bar['volume'] += volume
            
            if current_bar['volume'] >= volume_threshold:
                bars.append(current_bar.copy())
                current_bar = {
                    'open': None,
                    'high': -np.inf,
                    'low': np.inf,
                    'close': None,
                    'volume': 0,
                    'timestamp': None
                }
        
        if current_bar['volume'] > 0:
            bars.append(current_bar)
        
        if not bars:
            return pd.DataFrame()
        
        bars_df = pd.DataFrame(bars)
        bars_df['datetime'] = pd.to_datetime(bars_df['timestamp'], unit='ms')
        bars_df.set_index('datetime', inplace=True)
        
        return bars_df[['open', 'high', 'low', 'close', 'volume']]
    
    @staticmethod
    def create_dollar_bars(trades_df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
        """Create dollar bars based on cumulative dollar volume"""
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        bars = []
        current_bar = {
            'open': None,
            'high': -np.inf,
            'low': np.inf,
            'close': None,
            'volume': 0,
            'dollar_volume': 0,
            'timestamp': None
        }
        
        for _, trade in trades_df.iterrows():
            price = trade['price']
            volume = trade['sz']
            dollar_vol = price * volume
            timestamp = trade['time']
            
            if current_bar['open'] is None:
                current_bar['open'] = price
                current_bar['timestamp'] = timestamp
            
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price
            current_bar['volume'] += volume
            current_bar['dollar_volume'] += dollar_vol
            
            if current_bar['dollar_volume'] >= dollar_threshold:
                bars.append(current_bar.copy())
                current_bar = {
                    'open': None,
                    'high': -np.inf,
                    'low': np.inf,
                    'close': None,
                    'volume': 0,
                    'dollar_volume': 0,
                    'timestamp': None
                }
        
        if current_bar['dollar_volume'] > 0:
            bars.append(current_bar)
        
        if not bars:
            return pd.DataFrame()
        
        bars_df = pd.DataFrame(bars)
        bars_df['datetime'] = pd.to_datetime(bars_df['timestamp'], unit='ms')
        bars_df.set_index('datetime', inplace=True)
        
        return bars_df[['open', 'high', 'low', 'close', 'volume']]

def plot_bars(bars_df: pd.DataFrame, title: str, bar_type: str):
    """Plot bars using mplfinance"""
    if bars_df.empty:
        st.warning(f"No data available for {bar_type}")
        return
    
    # Configure plot style
    mc = mpf.make_marketcolors(
        up='g', down='r',
        edge='inherit',
        wick={'up':'green', 'down':'red'},
        volume='in'
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        y_on_right=True
    )
    
    # Create the plot
    fig, axes = mpf.plot(
        bars_df,
        type='candle',
        style=s,
        title=title,
        volume=True,
        figsize=(12, 8),
        returnfig=True
    )
    
    st.pyplot(fig)

def main():
    st.title("₿ Hyperliquid BTC Trading Bars Analysis")
    st.markdown("Analyze Bitcoin trades from Hyperliquid using different bar types")
    
    # Initialize API
    api = HyperliquidAPI()
    bar_builder = BarBuilder()
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    # Data fetching parameters
    st.sidebar.subheader("Data Parameters")
    max_trades = st.sidebar.slider("Maximum trades to fetch", 100, 5000, 1000, 100)
    
    # Bar type selection
    st.sidebar.subheader("Bar Type")
    bar_type = st.sidebar.selectbox(
        "Select bar type",
        ["Tick Bars", "Time Bars", "Volume Bars", "Dollar Bars"]
    )
    
    # Threshold parameters based on bar type
    st.sidebar.subheader("Threshold Parameters")
    
    if bar_type == "Tick Bars":
        threshold = st.sidebar.slider("Ticks per bar", 1, 200, 50)
    elif bar_type == "Time Bars":
        threshold = st.sidebar.slider("Minutes per bar", 1, 120, 15)
    elif bar_type == "Volume Bars":
        threshold = st.sidebar.number_input("Volume threshold", 0.01, 100.0, 1.0, 0.01)
    else:  # Dollar Bars
        threshold = st.sidebar.number_input("Dollar volume threshold", 100, 500000, 10000, 100)
    
    # Fetch data button
    if st.sidebar.button("Fetch & Analyze Data", type="primary"):
        with st.spinner("Fetching trades from Hyperliquid..."):
            trades = api.get_recent_trades("BTC", max_trades)
            
            if not trades:
                st.error("No trades data received")
                return
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Convert price and size to numeric
            trades_df['price'] = pd.to_numeric(trades_df['px'])
            trades_df['sz'] = pd.to_numeric(trades_df['sz'])
            trades_df['time'] = pd.to_numeric(trades_df['time'])
            
            # Sort by timestamp
            trades_df = trades_df.sort_values('time')
            
            st.success(f"Fetched {len(trades_df)} trades")
            
            # Display raw data statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Trades", len(trades_df))
            with col2:
                st.metric("Price Range", f"${trades_df['price'].min():.2f} - ${trades_df['price'].max():.2f}")
            with col3:
                st.metric("Total Volume", f"{trades_df['sz'].sum():.2f}")
            with col4:
                st.metric("Avg Trade Size", f"{trades_df['sz'].mean():.4f}")
            
            # Show suggested thresholds based on data
            st.info(f"""
            **Suggested thresholds based on your data:**
            - Tick Bars: {len(trades_df)//10} to {len(trades_df)//5} ticks per bar
            - Time Bars: {max(1, int((trades_df['time'].max() - trades_df['time'].min()) / (1000 * 60 * 10)))} to {max(1, int((trades_df['time'].max() - trades_df['time'].min()) / (1000 * 60 * 5)))} minutes per bar
            - Volume Bars: {trades_df['sz'].sum()/20:.3f} to {trades_df['sz'].sum()/10:.3f} volume per bar
            - Dollar Bars: ${(trades_df['price'] * trades_df['sz']).sum()/20:,.0f} to ${(trades_df['price'] * trades_df['sz']).sum()/10:,.0f} per bar
            """)
            
            # Create bars based on selected type
            with st.spinner(f"Creating {bar_type.lower()}..."):
                if bar_type == "Tick Bars":
                    bars_df = bar_builder.create_tick_bars(trades_df, threshold)
                    title = f"BTC Tick Bars ({threshold} ticks per bar)"
                elif bar_type == "Time Bars":
                    bars_df = bar_builder.create_time_bars(trades_df, threshold)
                    title = f"BTC Time Bars ({threshold} min per bar)"
                elif bar_type == "Volume Bars":
                    bars_df = bar_builder.create_volume_bars(trades_df, threshold)
                    title = f"BTC Volume Bars ({threshold} volume per bar)"
                else:  # Dollar Bars
                    bars_df = bar_builder.create_dollar_bars(trades_df, threshold)
                    title = f"BTC Dollar Bars (${threshold:,} per bar)"
                
                if not bars_df.empty:
                    st.subheader(title)
                    
                    # Display bar statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Bars", len(bars_df))
                    with col2:
                        st.metric("Avg Bar Volume", f"{bars_df['volume'].mean():.4f}")
                    with col3:
                        st.metric("Time Span", f"{(bars_df.index[-1] - bars_df.index[0]).total_seconds() / 3600:.1f}h")
                    
                    # Only plot if we have multiple bars
                    if len(bars_df) > 1:
                        # Plot the bars
                        plot_bars(bars_df, title, bar_type)
                    else:
                        st.warning("Only 1 bar created. Try using a smaller threshold to get more bars.")
                    
                    # Show recent bars data
                    with st.expander("Recent Bars Data"):
                        st.dataframe(bars_df.tail(10).round(4))
                else:
                    st.warning("No bars could be created with the current threshold. Try adjusting the parameters.")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Bar Types:**
    - **Tick Bars**: Fixed number of transactions
    - **Time Bars**: Fixed time intervals
    - **Volume Bars**: Fixed volume amounts
    - **Dollar Bars**: Fixed dollar volume amounts
    
    Data is fetched from Hyperliquid's public API.
    """)

if __name__ == "__main__":
    main()