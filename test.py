# %%
# financial_bars_analysis.py
# Jupyter-friendly script to build, compare, and plot Time, Tick, Volume, Dollar bars

# %%
# 1. Imports
import pandas as pd
import numpy as np
import mplfinance as mpf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
# 2. Load LOBSTER data (update filename and date as needed)
cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
df = pd.read_csv('aapl_lob.csv',
                 header=None, names=cols)
# Keep only trade events (types 4 & 5)
trades = df[df['type'].isin([4, 5])].copy()
# Convert price from integer to float dollars
trades['price'] = trades['price'] / 10000.0
trades = trades.sort_values('time').reset_index(drop=True)
# Compute dollar volume per trade
trades['dollar'] = trades['price'] * trades['size']

# Base date for plotting (year-month-day of the data)
base_date = '2025-06-14'

# %%
# 3. Bar-building functions

def get_time_bars(trades, interval_sec=60):
    trades = trades.copy()
    trades['bin'] = np.floor((trades['time'] - trades['time'].iloc[0]) / interval_sec)
    grouped = trades.groupby('bin', as_index=False)
    df_bar = grouped.agg(
        time=('time', 'first'),
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('size', 'sum'),
        dollar=('dollar', 'sum'),
        n_trades=('price', 'count'),
        duration=('time', lambda x: x.max() - x.min())
    )
    df_bar['volatility'] = df_bar['high'] - df_bar['low']
    return df_bar


def get_tick_bars(trades, tick_threshold=100):
    bars = []
    start = 0
    for i in range(len(trades)):
        if (i - start + 1) >= tick_threshold:
            seg = trades.iloc[start:i+1]
            bars.append({
                'time': seg['time'].iloc[-1],
                'open': seg['price'].iloc[0],
                'high': seg['price'].max(),
                'low': seg['price'].min(),
                'close': seg['price'].iloc[-1],
                'volume': seg['size'].sum(),
                'dollar': seg['dollar'].sum(),
                'n_trades': len(seg),
                'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
            })
            start = i + 1
    df_bar = pd.DataFrame(bars)
    df_bar['volatility'] = df_bar['high'] - df_bar['low']
    return df_bar


def get_volume_bars(trades, vol_threshold=7000):
    bars = []
    cum_vol = 0
    start = 0
    for i, row in trades.iterrows():
        cum_vol += row['size']
        if cum_vol >= vol_threshold:
            seg = trades.iloc[start:i+1]
            bars.append({
                'time': seg['time'].iloc[-1],
                'open': seg['price'].iloc[0],
                'high': seg['price'].max(),
                'low': seg['price'].min(),
                'close': seg['price'].iloc[-1],
                'volume': seg['size'].sum(),
                'dollar': seg['dollar'].sum(),
                'n_trades': len(seg),
                'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
            })
            start = i + 1
            cum_vol = 0
    df_bar = pd.DataFrame(bars)
    df_bar['volatility'] = df_bar['high'] - df_bar['low']
    return df_bar


def get_dollar_bars(trades, doll_threshold=4e6):
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
                'volume': seg['size'].sum(),
                'dollar': seg['dollar'].sum(),
                'n_trades': len(seg),
                'duration': seg['time'].iloc[-1] - seg['time'].iloc[0]
            })
            start = i + 1
            cum_doll = 0
    df_bar = pd.DataFrame(bars)
    df_bar['volatility'] = df_bar['high'] - df_bar['low']
    return df_bar

# %%
# 4. Build bars with chosen thresholds
interval_sec = 60
tick_threshold = 100
vol_threshold = 7000
doll_threshold = 4e6

time_bars   = get_time_bars(trades, interval_sec=interval_sec)
tick_bars   = get_tick_bars(trades, tick_threshold=tick_threshold)
volume_bars = get_volume_bars(trades, vol_threshold=vol_threshold)
dollar_bars = get_dollar_bars(trades, doll_threshold=doll_threshold)

# %%
# 5. Plotting function using mplfinance
def plot_bars(df, title):
    df_plot = df.set_index(pd.to_datetime(df['time'], unit='s', origin=pd.Timestamp(base_date)))
    mpf.plot(df_plot[['open','high','low','close','volume']],
             type='candle', volume=True, title=title)

# Generate plots
plot_bars(time_bars, 'Time Bars')
plot_bars(tick_bars, 'Tick Bars')
plot_bars(volume_bars, 'Volume Bars')
plot_bars(dollar_bars, 'Dollar Bars')

# %%
# 6. Correlation analysis between a trading feature and volatility
corrs = {
    'Time':   time_bars['n_trades'].corr(time_bars['volatility']),
    'Tick':   tick_bars['duration'].corr(tick_bars['volatility']),
    'Volume': volume_bars['duration'].corr(volume_bars['volatility']),
    'Dollar': dollar_bars['duration'].corr(dollar_bars['volatility'])
}
print("Correlation with volatility:", corrs)

# %%
# 7. Predictive modeling: sign of next bar return
def prepare_features(df, lags=5):
    df = df.copy()
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['return'].shift(lag)
    df = df.dropna()
    df['target'] = (df['return'] > 0).astype(int)
    X = df[[f'lag_{lag}' for lag in range(1, lags+1)]]
    y = df['target']
    return train_test_split(X, y, test_size=0.3, random_state=42)

results = {}
bar_dict = {
    'Time': time_bars,
    'Tick': tick_bars,
    'Volume': volume_bars,
    'Dollar': dollar_bars
}
for name, bars in bar_dict.items():
    X_train, X_test, y_train, y_test = prepare_features(bars)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = {
        'bar_count': len(bars),
        'correlation': corrs[name],
        'accuracy': acc
    }

# %%
# 8. Display results
results_df = pd.DataFrame(results).T
print(results_df)
