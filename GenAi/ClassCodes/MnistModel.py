import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

# Visualize one sample
plt.imshow(x_train[1], cmap='gray')
plt.show()

# Normalize and flatten
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print(x_train.shape)

# Build model
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(50, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(40, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print(model.summary())

# Compile & Train
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001),
              metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=20)

# Evaluate
model.evaluate(x_test, y_test, batch_size=32)
