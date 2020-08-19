import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import numpy
import random

stock = yf.Ticker(input("Stock Code : "))
history = stock.history(period = "max")

stock_prices = []
stock_index = []


for i in history["Close"]:
    prices.append(int(i))

for i in range(1,len(prices) + 1):
    index.append(i)

#building my own neural network cause why not

def sigmoid(z):
    return 1/(1 + numpy.exp(-z))

def intialize_constants(lim):
    bias = 0
    w = numpy.zeros(shape = (dim,1))

def propogate(w,b,x,y):
    m = x.shape[1]

    a = sigmoid(numpy.dot(w.t,x) + b)
    cost = (-1/m) * (np.sum(y * np.log(a) + (1-y) * (np.log(1-a))))

    dw = (1/m) * np.dot(x,(a-y).t)
    db = (1/m) * np.sum(a-y)

    assert(dw.shape == w.shape) 
    assert(db.dtypes == float)

    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw" : dw, "db" : db}

    return grads,cost

