import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import streamlit as st
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import time as t
import asyncio
import aiohttp
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

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
use_custom = st.sidebar.checkbox("Use custom symbol", key="use_custom_symbol")
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
use_cache = st.sidebar.checkbox("Use cached data", value=True, key="use_cache_data")
batch_size = st.sidebar.selectbox("Batch size", [500, 1000], index=1)
delay_ms = st.sidebar.slider("Delay between requests (ms)", 100, 2000, 500, 50)

# Prediction model options
st.sidebar.subheader("Prediction Models")
enable_predictions = st.sidebar.checkbox("Enable Price Movement Prediction", value=True, key="enable_predictions")

if enable_predictions:
    lookback_periods = st.sidebar.slider("Lookback periods for features", 3, 20, 10, 1)
    test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    min_bars_for_prediction = st.sidebar.number_input("Minimum bars for prediction", 50, 500, 100, 10)
    
    # Preheating options
    st.sidebar.subheader("Model Preheating")
    enable_preheating = st.sidebar.checkbox("Enable Model Preheating", value=True, key="enable_preheating")
    if enable_preheating:
        preheat_ratio = st.sidebar.slider("Preheating data ratio", 0.2, 0.6, 0.4, 0.05)
        walk_forward = st.sidebar.checkbox("Walk-forward validation", value=True, key="walk_forward_validation")
        refit_frequency = st.sidebar.selectbox("Model refit frequency", ["Never", "Every 10 bars", "Every 20 bars", "Every 50 bars"], index=1)
    
    # Real-time testing options
    st.sidebar.subheader("Real-Time Testing")
    enable_realtime = st.sidebar.checkbox("Enable Real-Time Prediction Testing", value=False, key="enable_realtime_testing")
    if enable_realtime:
        realtime_interval = st.sidebar.selectbox("Update interval", ["30 seconds", "1 minute", "2 minutes", "5 minutes"], index=1)
        max_realtime_predictions = st.sidebar.number_input("Max predictions to track", 10, 100, 50, 5)
    
    # Feature selection options
    st.sidebar.subheader("Feature Selection")
    enable_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=True, key="enable_feature_selection")
    if enable_feature_selection:
        feature_selection_methods = st.sidebar.multiselect(
            "Selection methods",
            ["Variance Threshold", "Univariate (f_classif)", "Mutual Information", "RFE", "L1 Regularization", "Tree Importance"],
            default=["Variance Threshold", "Mutual Information", "Tree Importance"],
            key="feature_selection_methods"
        )
        max_features_ratio = st.sidebar.slider("Max features to keep (ratio)", 0.1, 1.0, 0.5, 0.1)
        min_feature_importance = st.sidebar.slider("Min feature importance threshold", 0.001, 0.1, 0.01, 0.001)
    
    models_to_use = st.sidebar.multiselect(
        "Select models to compare",
        ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "Neural Network"],
        default=["Logistic Regression", "Random Forest", "Gradient Boosting"],
        key="models_to_use"
    )

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

# Fetch trades
st.header(f"📈 {selected_crypto} Trading Bars Analysis")

col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"Enhanced analysis for **{selected_symbol}** with better rate limiting and larger data fetching capabilities")

with st.spinner(f"Fetching {selected_symbol} trades from Binance..."):
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
    
    # Add download buttons
    st.subheader("📥 Download Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download raw trades data
        trades_csv = trades.to_csv(index=False)
        st.download_button(
            label="📊 Download Raw Trades Data (CSV)",
            data=trades_csv,
            file_name=f"{selected_symbol}_trades_{start_time.strftime('%Y%m%d_%H%M')}_to_{end_time.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download the raw trade-by-trade data"
        )
    
    with col2:
        # Preview trades data
        if st.button("👁️ Preview Raw Trades Data"):
            st.write("**Sample of Raw Trades Data:**")
            st.dataframe(trades.head(10), use_container_width=True)
else:
    st.error("No trades were fetched.")
    st.stop()

# Bar-building functions
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

# Advanced feature engineering for time series prediction
class TimeSeriesFeatureEngineer:
    def __init__(self, lookback_periods=10):
        self.lookback_periods = lookback_periods
        
    def create_features(self, bars_df):
        """Create comprehensive features for time series prediction"""
        df = bars_df.copy()
        
        # Basic price features
        df['return'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Volatility features
        df['volatility_norm'] = df['volatility'] / df['close']
        df['volume_price_trend'] = df['volume'] * df['return']
        
        # Rolling statistics
        for window in [3, 5, 10]:
            df[f'return_ma_{window}'] = df['return'].rolling(window).mean()
            df[f'return_std_{window}'] = df['return'].rolling(window).std()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volatility_ma_{window}'] = df['volatility'].rolling(window).mean()
        
        # Momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        
        # Lagged features
        feature_cols = ['return', 'price_change', 'volatility_norm', 'volume_change', 'rsi']
        for col in feature_cols:
            for lag in range(1, self.lookback_periods + 1):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Target variable (next period return sign)
        df['target'] = (df['return'].shift(-1) > 0).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

class AdvancedFeatureSelector:
    def __init__(self, selection_methods, max_features_ratio=0.5, min_importance=0.01):
        self.selection_methods = selection_methods
        self.max_features_ratio = max_features_ratio
        self.min_importance = min_importance
        self.selected_features = []
        self.feature_scores = {}
        
    def remove_low_variance_features(self, X, threshold=0.01):
        """Remove features with low variance"""
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        
        variances = selector.variances_
        feature_scores = dict(zip(X.columns, variances))
        
        return X_selected, selected_features, feature_scores
    
    def univariate_selection(self, X, y, k='auto'):
        """Select features based on univariate statistical tests"""
        if k == 'auto':
            k = max(5, min(int(len(X.columns) * self.max_features_ratio), len(X.columns)))
        
        # F-classification test
        selector_f = SelectKBest(score_func=f_classif, k=k)
        X_selected_f = selector_f.fit_transform(X, y)
        selected_features_f = X.columns[selector_f.get_support()].tolist()
        f_scores = dict(zip(X.columns, selector_f.scores_))
        
        return X_selected_f, selected_features_f, f_scores
    
    def mutual_information_selection(self, X, y, k='auto'):
        """Select features based on mutual information"""
        if k == 'auto':
            k = max(5, min(int(len(X.columns) * self.max_features_ratio), len(X.columns)))
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        mi_scores = dict(zip(X.columns, selector.scores_))
        
        return X_selected, selected_features, mi_scores
    
    def rfe_selection(self, X, y, estimator=None):
        """Recursive Feature Elimination"""
        if estimator is None:
            estimator = LogisticRegression(random_state=42, max_iter=1000)
        
        n_features = max(5, int(len(X.columns) * self.max_features_ratio))
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature rankings (lower is better)
        rankings = dict(zip(X.columns, selector.ranking_))
        
        return X_selected, selected_features, rankings
    
    def l1_regularization_selection(self, X, y, C=0.1):
        """L1 regularization for feature selection"""
        estimator = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
        selector = SelectFromModel(estimator, threshold=self.min_importance)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature coefficients
        estimator.fit(X, y)
        coefficients = dict(zip(X.columns, np.abs(estimator.coef_[0])))
        
        return X_selected, selected_features, coefficients
    
    def tree_importance_selection(self, X, y):
        """Tree-based feature importance"""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        estimator.fit(X, y)
        
        selector = SelectFromModel(estimator, threshold=self.min_importance)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature importances
        importances = dict(zip(X.columns, estimator.feature_importances_))
        
        return X_selected, selected_features, importances
    
    def select_features(self, X, y, feature_names):
        """Apply multiple feature selection methods and combine results"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        all_selected_features = set()
        method_results = {}
        
        # Apply each selected method
        for method in self.selection_methods:
            try:
                if method == "Variance Threshold":
                    X_sel, features, scores = self.remove_low_variance_features(X)
                elif method == "Univariate (f_classif)":
                    X_sel, features, scores = self.univariate_selection(X, y)
                elif method == "Mutual Information":
                    X_sel, features, scores = self.mutual_information_selection(X, y)
                elif method == "RFE":
                    X_sel, features, scores = self.rfe_selection(X, y)
                elif method == "L1 Regularization":
                    X_sel, features, scores = self.l1_regularization_selection(X, y)
                elif method == "Tree Importance":
                    X_sel, features, scores = self.tree_importance_selection(X, y)
                
                all_selected_features.update(features)
                method_results[method] = {
                    'features': features,
                    'scores': scores,
                    'n_selected': len(features)
                }
                
            except Exception as e:
                st.warning(f"Feature selection method {method} failed: {e}")
                continue
        
        # Combine results - use intersection or union based on strategy
        if len(self.selection_methods) > 1:
            # Use features selected by at least 2 methods (intersection approach)
            feature_votes = {}
            for method, result in method_results.items():
                for feature in result['features']:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            # Select features that appear in multiple methods
            min_votes = max(1, len(self.selection_methods) // 2)
            final_selected_features = [f for f, votes in feature_votes.items() if votes >= min_votes]
        else:
            final_selected_features = list(all_selected_features)
        
        # Ensure we don't have too many features
        max_features = int(len(X.columns) * self.max_features_ratio)
        if len(final_selected_features) > max_features:
            # If we have too many, rank by average importance across methods
            feature_avg_scores = {}
            for feature in final_selected_features:
                scores = []
                for method, result in method_results.items():
                    if feature in result['features'] and feature in result['scores']:
                        # Normalize scores to 0-1 range for comparison
                        method_scores = list(result['scores'].values())
                        max_score = max(method_scores) if method_scores else 1
                        normalized_score = result['scores'][feature] / max_score if max_score > 0 else 0
                        scores.append(normalized_score)
                
                feature_avg_scores[feature] = np.mean(scores) if scores else 0
            
            # Select top features by average score
            sorted_features = sorted(feature_avg_scores.items(), key=lambda x: x[1], reverse=True)
            final_selected_features = [f[0] for f in sorted_features[:max_features]]
        
        # Store results
        self.selected_features = final_selected_features
        self.feature_scores = method_results
        
        # Return selected feature matrix
        X_final = X[final_selected_features]
        
        return X_final, final_selected_features, method_results

class TimeSeriesPredictor:
    def __init__(self, models_to_use, test_size=0.2, random_state=42):
        self.models_to_use = models_to_use
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Initialize models
        self.models = {}
        if "Logistic Regression" in models_to_use:
            self.models["Logistic Regression"] = LogisticRegression(random_state=random_state, max_iter=1000)
        if "Random Forest" in models_to_use:
            self.models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
        if "Gradient Boosting" in models_to_use:
            self.models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=random_state, max_depth=5)
        if "SVM" in models_to_use:
            self.models["SVM"] = SVC(random_state=random_state, probability=True, kernel='rbf')
        if "Neural Network" in models_to_use:
            self.models["Neural Network"] = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=500)
    
    def prepare_data(self, df_with_features, feature_selector=None):
        """Prepare data for modeling with optional feature selection"""
        # Remove rows with NaN values
        df_clean = df_with_features.dropna()
        
        if len(df_clean) < 50:
            return None, None, None, None
        
        # Select feature columns (exclude target and time-related columns)
        feature_cols = [col for col in df_clean.columns if 
                       col not in ['time', 'target', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility', 'bin']]
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Apply feature selection if provided
        if feature_selector is not None:
            X_selected, selected_features, selection_results = feature_selector.select_features(X, y, feature_cols)
            return X_selected, y, selected_features, selection_results
        
        return X, y, feature_cols, None
    
    def walk_forward_validation(self, X, y, preheat_ratio=0.4, refit_frequency="Every 10 bars"):
        """
        Perform walk-forward validation with model preheating
        """
        n_samples = len(X)
        preheat_size = int(n_samples * preheat_ratio)
        
        # Parse refit frequency
        refit_freq_map = {
            "Never": np.inf,
            "Every 10 bars": 10,
            "Every 20 bars": 20, 
            "Every 50 bars": 50
        }
        refit_freq = refit_freq_map[refit_frequency]
        
        results = {}
        
        for name, base_model in self.models.items():
            st.write(f"Running walk-forward validation for {name}...")
            
            predictions = []
            probabilities = []
            actual_values = []
            preheat_accuracies = []
            
            # Initialize model with preheating data
            X_preheat = X.iloc[:preheat_size]
            y_preheat = y.iloc[:preheat_size]
            
            # Scale preheating data
            scaler = StandardScaler()
            X_preheat_scaled = scaler.fit_transform(X_preheat)
            
            # Clone and train model on preheating data
            from sklearn.base import clone
            model = clone(base_model)
            model.fit(X_preheat_scaled, y_preheat)
            
            # Calculate preheating accuracy
            preheat_pred = model.predict(X_preheat_scaled)
            preheat_acc = accuracy_score(y_preheat, preheat_pred)
            preheat_accuracies.append(preheat_acc)
            
            # Walk forward through remaining data
            for i in range(preheat_size, n_samples):
                # Current observation
                X_current = X.iloc[[i]]
                y_current = y.iloc[i]
                
                # Scale current observation using existing scaler
                X_current_scaled = scaler.transform(X_current)
                
                # Make prediction
                pred = model.predict(X_current_scaled)[0]
                predictions.append(pred)
                actual_values.append(y_current)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_current_scaled)[0, 1]
                    probabilities.append(prob)
                
                # Refit model if needed
                if (i - preheat_size + 1) % refit_freq == 0 and refit_freq != np.inf:
                    # Use expanding window for retraining
                    X_retrain = X.iloc[:i+1]
                    y_retrain = y.iloc[:i+1]
                    
                    # Refit scaler and model
                    X_retrain_scaled = scaler.fit_transform(X_retrain)
                    model = clone(base_model)
                    model.fit(X_retrain_scaled, y_retrain)
            
            # Calculate metrics
            if predictions:
                accuracy = accuracy_score(actual_values, predictions)
                precision = precision_score(actual_values, predictions, average='weighted', zero_division=0)
                recall = recall_score(actual_values, predictions, average='weighted', zero_division=0)
                f1 = f1_score(actual_values, predictions, average='weighted', zero_division=0)
                baseline_accuracy = max(np.mean(actual_values), 1 - np.mean(actual_values))
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement_over_baseline': accuracy - baseline_accuracy,
                    'predictions': predictions,
                    'probabilities': probabilities if probabilities else None,
                    'actual': actual_values,
                    'preheat_accuracy': preheat_acc,
                    'preheat_size': preheat_size,
                    'prediction_count': len(predictions),
                    'refit_frequency': refit_frequency
                }
            else:
                results[name] = None
                
        return results
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance (original method for comparison)"""
        results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Calculate directional accuracy (most important for trading)
                baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())  # Majority class accuracy
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'baseline_accuracy': baseline_accuracy,
                    'improvement_over_baseline': accuracy - baseline_accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'actual': y_test
                }
                
            except Exception as e:
                st.warning(f"Error training {name}: {e}")
                results[name] = None
        
        return results

def run_prediction_analysis(bars_dict, feature_engineer, predictor, min_bars, enable_preheating, preheat_ratio, walk_forward, refit_frequency, feature_selector=None):
    """Run prediction analysis for all bar types with optional feature selection"""
    prediction_results = {}
    feature_selection_results = {}
    
    for bar_type, bars_df in bars_dict.items():
        if len(bars_df) < min_bars:
            st.warning(f"Skipping {bar_type} bars: only {len(bars_df)} bars (need at least {min_bars})")
            continue
        
        st.write(f"**Analyzing {bar_type} Bars ({len(bars_df)} bars)...**")
        
        # Feature engineering
        df_with_features = feature_engineer.create_features(bars_df)
        
        # Prepare data with optional feature selection
        if feature_selector is not None:
            X, y, feature_cols, selection_results = predictor.prepare_data(df_with_features, feature_selector)
            feature_selection_results[bar_type] = selection_results
            
            if selection_results:
                original_feature_count = len([col for col in df_with_features.columns if col not in ['time', 'target', 'open', 'high', 'low', 'close', 'volume', 'dollar', 'n_trades', 'duration', 'volatility', 'bin']])
                st.info(f"Feature selection for {bar_type}: {len(feature_cols)} features selected from {original_feature_count} original features")
        else:
            X, y, feature_cols, _ = predictor.prepare_data(df_with_features)
        
        if X is None:
            st.warning(f"Insufficient clean data for {bar_type} bars after feature engineering")
            continue
        
        # Choose analysis method
        if enable_preheating and walk_forward:
            st.info(f"Using walk-forward validation with {preheat_ratio:.0%} preheating for {bar_type} bars")
            results = predictor.walk_forward_validation(X, y, preheat_ratio, refit_frequency)
        else:
            # Traditional train/test split
            split_point = int(len(X) * (1 - predictor.test_size))
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if enable_preheating:
                # Simple preheating: use first portion for initial training
                preheat_size = int(len(X_train) * preheat_ratio)
                st.info(f"Preheating models with {preheat_size} bars for {bar_type}")
                # In this case, we just use the preheat portion as training data
                X_train_scaled = X_train_scaled[:preheat_size]
                y_train = y_train.iloc[:preheat_size]
            
            results = predictor.train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Add feature count to results
            for model_name in results:
                if results[model_name] is not None:
                    results[model_name]['n_features'] = len(feature_cols)
                    results[model_name]['model_source'] = 'trained'
        
        prediction_results[bar_type] = {
            'results': results,
            'n_total': len(X),
            'feature_count': len(feature_cols),
            'class_distribution': y.value_counts().to_dict(),
            'analysis_method': 'Walk-Forward' if (enable_preheating and walk_forward) else 'Traditional Split'
        }
    
    return prediction_results, feature_selection_results

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

# Add download buttons for bar data
st.subheader("📥 Download Bar Data")
download_cols = st.columns(len(bars_dict))

for i, (name, bars) in enumerate(bars_dict.items()):
    with download_cols[i]:
        if len(bars) > 0:
            bars_csv = bars.to_csv(index=False)
            st.download_button(
                label=f"📈 {name} Bars",
                data=bars_csv,
                file_name=f"{selected_symbol}_{name.lower()}_bars_{start_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help=f"Download {name} bars data ({len(bars)} bars)"
            )
        else:
            st.write(f"No {name} bars")

# Option to download all bars in one file
if any(len(bars) > 0 for bars in bars_dict.values()):
    with st.expander("📦 Download All Bar Types Combined"):
        # Combine all bar types into one DataFrame
        combined_bars = []
        for bar_type, bars in bars_dict.items():
            if len(bars) > 0:
                bars_with_type = bars.copy()
                bars_with_type['bar_type'] = bar_type
                combined_bars.append(bars_with_type)
        
        if combined_bars:
            all_bars_df = pd.concat(combined_bars, ignore_index=True)
            all_bars_csv = all_bars_df.to_csv(index=False)
            
            st.download_button(
                label="📦 Download All Bar Types (Combined CSV)",
                data=all_bars_csv,
                file_name=f"{selected_symbol}_all_bars_{start_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download all bar types in a single CSV file with bar_type column"
            )

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
                
                # Add individual download button for this bar type
                bars_csv = bars.to_csv(index=False)
                st.download_button(
                    label=f"📊 Download {name} Bars CSV",
                    data=bars_csv,
                    file_name=f"{selected_symbol}_{name.lower()}_bars_detailed_{bars['time'].min().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key=f"download_{name}_bars_detailed"
                )
                
        except Exception as e:
            st.error(f"Error plotting {name} bars: {e}")

# Correlation & predictive model
st.header("Statistical Analysis & Price Movement Prediction")

def run_basic_correlation_model(bars):
    """Basic correlation analysis (original function)"""
    if len(bars) < 20:
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

# Calculate basic correlations and model accuracy
basic_results = {}
for name, bars in bars_dict.items():
    if len(bars) > 1:
        # Correlation between duration and volatility (for time bars) or n_trades and volatility (for others)
        if name == 'Time':
            corr = bars['duration'].corr(bars['volatility'])
        else:
            corr = bars['n_trades'].corr(bars['volatility'])
        
        accuracy = run_basic_correlation_model(bars)
        
        basic_results[name] = {
            'Correlation (Duration/Trades vs Volatility)': corr,
            'Basic Model Accuracy': accuracy,
            'Number of Bars': len(bars)
        }
    else:
        basic_results[name] = {
            'Correlation (Duration/Trades vs Volatility)': np.nan,
            'Basic Model Accuracy': np.nan,
            'Number of Bars': len(bars)
        }

basic_results_df = pd.DataFrame(basic_results).T
st.subheader("Basic Statistical Analysis")
st.dataframe(basic_results_df.round(4), use_container_width=True)

# Advanced Time Series Prediction Analysis
if enable_predictions:
    st.subheader("🔮 Advanced Time Series Prediction Analysis")
    
    # Check if we have enough data
    suitable_bars = {name: bars for name, bars in bars_dict.items() if len(bars) >= min_bars_for_prediction}
    
    if not suitable_bars:
        st.warning(f"No bar types have enough data for prediction (need at least {min_bars_for_prediction} bars)")
    else:
        with st.spinner("Running advanced time series prediction analysis..."):
            # Initialize feature engineer and predictor
            feature_engineer = TimeSeriesFeatureEngineer(lookback_periods=lookback_periods)
            predictor = TimeSeriesPredictor(models_to_use, test_size=test_size)
            
            # Initialize feature selector if enabled
            feature_selector = None
            if enable_feature_selection:
                feature_selector = AdvancedFeatureSelector(
                    selection_methods=feature_selection_methods,
                    max_features_ratio=max_features_ratio,
                    min_importance=min_feature_importance
                )
                st.info(f"Feature selection enabled: {', '.join(feature_selection_methods)}")
            
            # Run prediction analysis with feature selection
            prediction_results, feature_selection_results = run_prediction_analysis(
                suitable_bars, feature_engineer, predictor, min_bars_for_prediction,
                enable_preheating, preheat_ratio if enable_preheating else 0.4,
                walk_forward if enable_preheating else False,
                refit_frequency if enable_preheating else "Never",
                feature_selector
            )
            
            # Display feature selection results
            if enable_feature_selection and feature_selection_results:
                st.subheader("🎯 Feature Selection Results")
                
                for bar_type, selection_results in feature_selection_results.items():
                    if selection_results:
                        with st.expander(f"Feature Selection Details: {bar_type} Bars"):
                            
                            # Show summary by method
                            for method, result in selection_results.items():
                                st.write(f"**{method}**: {result['n_selected']} features selected")
                                
                                # Show top features for this method
                                if result['scores']:
                                    top_features = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)[:10]
                                    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Score'])
                                    st.dataframe(feature_df, use_container_width=True)
                            
                            # Show final selected features
                            if feature_selector and hasattr(feature_selector, 'selected_features'):
                                st.write(f"**Final Selected Features ({len(feature_selector.selected_features)}):**")
                                selected_df = pd.DataFrame({'Feature': feature_selector.selected_features})
                                st.dataframe(selected_df, use_container_width=True)

            if prediction_results:
                # Create comparison table
                comparison_data = []
                for bar_type, result in prediction_results.items():
                    if result and result['results']:
                        for model_name, model_result in result['results'].items():
                            if model_result:
                                comparison_data.append({
                                    'Bar Type': bar_type,
                                    'Model': model_name,
                                    'Analysis Method': result['analysis_method'],
                                    'Features Used': model_result.get('n_features', result['feature_count']),
                                    'Accuracy': model_result['accuracy'],
                                    'Precision': model_result['precision'],
                                    'Recall': model_result['recall'],
                                    'F1 Score': model_result['f1_score'],
                                    'Baseline Accuracy': model_result['baseline_accuracy'],
                                    'Improvement over Baseline': model_result['improvement_over_baseline'],
                                    'Preheat Accuracy': model_result.get('preheat_accuracy', 'N/A'),
                                    'Preheat Size': model_result.get('preheat_size', 'N/A'),
                                    'Predictions Made': model_result.get('prediction_count', result.get('n_test', 'N/A')),
                                    'Refit Frequency': model_result.get('refit_frequency', 'N/A')
                                })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display results
                    st.subheader("📊 Model Performance Comparison")
                    
                    # Color-code the dataframe
                    def highlight_best_performance(s):
                        if s.name in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Improvement over Baseline']:
                            max_val = s.max()
                            return ['background-color: lightgreen' if v == max_val else '' for v in s]
                        return ['' for _ in s]
                    
                    styled_df = comparison_df.round(4).style.apply(highlight_best_performance)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Add download button for prediction results
                    st.subheader("📥 Download Prediction Results")
                    prediction_csv = comparison_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Model Comparison Results (CSV)",
                        data=prediction_csv,
                        file_name=f"{selected_symbol}_prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download complete model performance comparison"
                    )
                    
                    # Best performing combinations
                    st.subheader("🏆 Top Performers")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Best by Accuracy:**")
                        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
                        st.success(f"{best_accuracy['Bar Type']} + {best_accuracy['Model']}: {best_accuracy['Accuracy']:.4f}")
                        
                        st.write("**Best Improvement over Baseline:**")
                        best_improvement = comparison_df.loc[comparison_df['Improvement over Baseline'].idxmax()]
                        st.success(f"{best_improvement['Bar Type']} + {best_improvement['Model']}: +{best_improvement['Improvement over Baseline']:.4f}")
                    
                    with col2:
                        st.write("**Best by F1 Score:**")
                        best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
                        st.success(f"{best_f1['Bar Type']} + {best_f1['Model']}: {best_f1['F1 Score']:.4f}")
                        
                        st.write("**Most Consistent (lowest std of accuracy):**")
                        accuracy_by_bar = comparison_df.groupby('Bar Type')['Accuracy'].agg(['mean', 'std'])
                        most_consistent = accuracy_by_bar.loc[accuracy_by_bar['std'].idxmin()]
                        st.success(f"{accuracy_by_bar['std'].idxmin()}: {most_consistent['mean']:.4f} (±{most_consistent['std']:.4f})")
                    
                    # Key insights and recommendations
                    st.subheader("🧠 Key Insights & Recommendations")
                    
                    # Find best bar type
                    bar_performance = comparison_df.groupby('Bar Type')['Accuracy'].mean().sort_values(ascending=False)
                    best_bar_type = bar_performance.index[0]
                    best_bar_accuracy = bar_performance.iloc[0]
                    
                    insights = []
                    insights.append(f"**Best Bar Type**: {best_bar_type} bars achieve the highest average accuracy ({best_bar_accuracy:.4f})")
                    
                    # Feature selection effectiveness
                    if enable_feature_selection and 'Features Used' in comparison_df.columns:
                        avg_features = comparison_df['Features Used'].apply(lambda x: x if isinstance(x, (int, float)) else 0).mean()
                        insights.append(f"**Feature Selection**: Reduced features to average of {avg_features:.1f} per model")
                    
                    # Check if any combination beats baseline significantly
                    significant_improvements = comparison_df[comparison_df['Improvement over Baseline'] > 0.05]
                    if len(significant_improvements) > 0:
                        best_combo = significant_improvements.loc[significant_improvements['Improvement over Baseline'].idxmax()]
                        insights.append(f"**Significant Improvement**: {best_combo['Bar Type']} + {best_combo['Model']} shows {best_combo['Improvement over Baseline']:.4f} improvement over baseline")
                    else:
                        insights.append("**Baseline Challenge**: No model-bar combination significantly outperforms the baseline (>5% improvement)")
                    
                    for insight in insights:
                        st.info(insight)
                
                else:
                    st.warning("No successful predictions were generated. Try adjusting the parameters or using more data.")
            else:
                st.warning("No suitable data for prediction analysis.")

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
- Consider fetching data in multiple sessions

**Rate Limit Info:**
- Binance: 1200 requests/minute
- Weight per request: 1
- Use delays of 50-100ms minimum
""")