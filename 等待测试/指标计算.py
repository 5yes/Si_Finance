import numpy as np
import pandas as pd


# 数据导入
def cal_main(path: str, variety: str):
    cal_data = pd.read_csv(path)
    cal_data = cal_data[cal_data['variety'] == variety]
    cal_data = cal_data.sort_values(['date', 'volume'], ascending=[True, False])
    cal_data = cal_data.drop_duplicates('date', keep='first')
    cal_data = cal_data.reset_index(drop=True)
    return cal_data


# MCDA
def cal_macd(df, short=12, long=26, signal=9):
    df['EMA_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    return df


# MTM
def cal_mtm(df, period=10):
    df['MTM'] = df['close'] - df['close'].shift(period)
    return df


# RSI
def cal_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (delta.where(delta < 0, 0).abs()).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


# BIAS
def cal_bias(df, period=6):
    df['MA'] = df['close'].rolling(window=period).mean()
    df['BIAS'] = (df['close'] - df['MA']) / df['MA'] * 100
    return df


# ROC
def cal_roc(df, period=12):
    df['ROC'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    return df


# MA
def cal_ma(df, period=5):
    df['MA'] = df['close'].rolling(window=period).mean()
    return df


# WR
def cal_wr(df, period=14):
    high = df['high'].rolling(window=period).max()
    low = df['low'].rolling(window=period).min()
    df['WR'] = (high - df['close']) / (high - low) * 100
    return df


# CCI
def cal_cci(df, period=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma) / (0.015 * mean_deviation)
    return df


# KDJ
def cal_kdj(df, period=9, k_period=3, d_period=3):
    low = df['low'].rolling(window=period).min()
    high = df['high'].rolling(window=period).max()
    df['RSV'] = (df['close'] - low) / (high - low) * 100
    df['K'] = df['RSV'].rolling(window=k_period).mean()
    df['D'] = df['K'].rolling(window=d_period).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df