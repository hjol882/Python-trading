import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Download data from a stock
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')

# Calculate Simple Moving Average (SMA)
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Calculate Exponential Moving Average (EMA)
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

# Calculate the Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data)

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    
    return macd, signal_line

data['MACD'], data['Signal_Line'] = calculate_macd(data)

# Plot the results
plt.figure(figsize=(14, 10))

# Charting prices and SMAs
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Precio de Cierre')
plt.plot(data['SMA_50'], label='SMA 50')
plt.plot(data['SMA_200'], label='SMA 200')
plt.title(f'Precio de Cierre y SMAs de {symbol}')
plt.legend()

# Chart RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label='RSI')
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.title('RSI')
plt.legend()

# Graficar MACD
plt.subplot(3, 1, 3)
plt.plot(data['MACD'], label='MACD')
plt.plot(data['Signal_Line'], label='Línea de Señal')
plt.title('MACD')
plt.legend()

plt.tight_layout()
plt.show()