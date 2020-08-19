import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import numpy

stock = yf.Ticker(input("Stock Code : "))
history = stock.history(period = "max")

stock_prices = []
stock_index = []


for i in history["Close"]:
    prices.append(int(i))

for i in range(1,len(prices) + 1):
    index.append(i)


