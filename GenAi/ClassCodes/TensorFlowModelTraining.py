import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y):
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  x_in = np.c_[xx.ravel(), yy.ravel()]
  y_pred = model.predict(x_in)
  if model.output_shape[-1] > 1:
    print("doing multiclass classification...")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

from sklearn.model_selection import train_test_split
X, y = make_circles(1000, noise=0.03)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)
history = model.fit(x_train, y_train, batch_size=80, epochs=100, validation_data=(x_test, y_test))

model.evaluate(x_test, y_test)
ypred = model.predict(x_test)
plot_decision_boundary(model, x_train, y_train)
plot_decision_boundary(model, x_test, y_test)