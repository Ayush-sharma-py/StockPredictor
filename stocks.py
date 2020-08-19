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

def optimize(w,b,x,y,num_iterations,learning_rate,print_cost = False):
    cost = []

    for i in renge(num_iterations):
        grads,cost = propogate(w,b,x,y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if i % 100 == 0 and print_cost:
            print("Cost after iteration {} is {}".format(i,cost))

    params = {"w" : w, "b" : b}
    grads = {"dw" : dw, "db" : db}

    return params,grads,costs

def predict(w,b,x):
    m = x.shape[1]
    y_prediction = numpy.zeros((1,m))
    w = w.reshape(x.shape[0],1)

    a = sigmoid(numpy.dot(w.t,x) + b)

    for i in range(a.shape[1]):
        y_prediction[0,i] = l if a[0,i] > 0.5 else 0 

    assert(y_prediction.shape == (1,m))
    return y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    #Initialize parameters with 0s
    w, b = initialize_with_zeros(X_train.shape[0])
    
    #Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    #Retrive parameters w, b from dictionary
    w = parameters['w']
    b = parameters['b']
    
    #Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    #Print test/train errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {'costs': costs,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}
    
    return d

        