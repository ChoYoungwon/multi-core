import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf

stock_data = yf.download('AAPL', start='2020-01-01', end='2025-04-04')

closing_prices = stock_data['Close']

price_fft = np.fft.fft(closing_prices)

frequencies = np.fft.fftfreq(len(closing_prices))

magnitude = np.abs(price_fft)

plt.figure(figsize=(10,6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
plt.title('Forier Transform - Stock Price Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.savefig('apple_fft')

predicted_prices = np.fft.ifft(price_fft)

plt.figure(figsize=(10,6))
plt.plot(closing_prices.index, closing_prices, label='Actual Prices')
plt.plot(closing_prices.index, predicted_prices.real, label='Predicted Prices', linestyle='--')
plt.legend()
plt.title('Stock Prices Prediction using Fourier Transform')

plt.xlim(closing_prices.index[0], closing_prices.index[-1])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.savefig('apple_original')