#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import yfinance as yf
import matplotlib

tsla = yf.Ticker("TSLA")
history = tsla.history(period = "max")
prices = []
for i in history["Close"]:
    prices.append(int(i))
print(prices)