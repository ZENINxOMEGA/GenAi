import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
X,y = make_circles(1000,noise=0.03)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()