import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import imdb
from keras import layers
from keras.preprocessing import sequence
top_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
print(x_train.shape,x_test.shape)
m = 0
for i in range(x_train.shape[0]):
    # print(len(x_train[i]))
    m = max(len(x_train[i]),m)
print(m)
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
x_train.shape

embedding_vector_length = 100
model = keras.Sequential()
model.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(layers.SimpleRNN(100,dropout=0.2,recurrent_dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

scores = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', scores[1])