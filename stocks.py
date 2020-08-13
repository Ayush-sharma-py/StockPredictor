import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
import numpy

tsla = yf.Ticker("TSLA")
history = tsla.history(period = "max")

prices = []
index = []


for i in history["Close"]:
    prices.append(int(i))

for i in range(1,len(prices) + 1):
    index.append(i)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
prices = numpy.array(prices)
index = numpy.array(index)
'''
model.fit(prices,index, epochs=10)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(index[-1] + 1)
