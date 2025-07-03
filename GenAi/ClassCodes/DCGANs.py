import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
import os

x = np.random.rand(4, 10, 8, 128)
y = keras.layers.Conv2DTranspose(32, 2, 2, activation='relu',)(x)
print(y.shape)

def discriminator(in_shape = (28,28,1)):
    model  = Sequential()
    model.add(keras.Input(shape=in_shape))
    
    model.add(keras.layers.Conv2D(64,3,2,padding='same',activation='leaky_relu'))
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Conv2D(128,3,2,padding='same',activation='leaky_relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(256,activation='leaky_relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics = ['accuracy'])

    return model

def generator(rv_dim = 100):
    model = Sequential()
    model.add(keras.Input(shape=(rv_dim,)))
    model.add(keras.layers.Dense(7*7*256,activation='leaky_relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Reshape((7,7,256)))

    model.add(keras.layers.Conv2DTranspose(128,3,2,padding='same',use_bias=False,activation = 'leaky_relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2DTranspose(64,3,2,padding='same',use_bias=False,activation = 'leaky_relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2DTranspose(1,3,1,padding='same',use_bias=False,activation = 'tanh'))

    model.compile(loss = 'mse',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    return model

def gan_model(g_model,d_model):
    model = Sequential()
    model.add(g_model)
    model.add(d_model)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics = ['accuracy'])
    return model

def load_real_data():
    (X_train,_),(_,_) = load_data()
    X_train = X_train.reshape((-1,28,28,1)).astype('f8')-127.5
    return X_train/127.5

def generate_real_samples(data,n_samples = 100):
    ix = np.random.randint(0,data.shape[0],n_samples)
    X_train = data[ix]
    y = np.ones(shape=(X_train.shape[0],1))
    return X_train,y

def generate_rv(rv_dim,n_sample=100):
    return np.random.randn(n_sample,rv_dim)

def generate_fake_images(g_model,rv_dim,n_samples = 100):
    rv = generate_rv(rv_dim,n_samples)
    fimg = g_model.predict(rv)
    y = np.zeros(shape = (n_samples,1))
    return fimg,y

def save_fig(g_model,rv_dim,epoch):
    n = 10
    rv = generate_rv(rv_dim,n*n)
    f_imgs = g_model.predict(rv)
    os.makedirs('./DCGAN_OUTPUT', exist_ok=True)
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow(f_imgs[i].reshape((28,28)), interpolation='nearest',cmap = 'gray')
    filename = f'./DCGAN_OUTPUT/generated_plot_e{epoch}.png'
    plt.savefig(filename)
    plt.close()

def train(data,g_model,d_model,gan_model,rv_dim,epochs = 51,batch_size = 256):
    nbatchs = data.shape[0]//batch_size
    half_batch = batch_size//2

    for e in range(epochs):
        for bn in range(nbatchs):
            x_real,y_real = generate_real_samples(data,half_batch)
            x_fake,y_fake = generate_fake_images(g_model,rv_dim,half_batch)

            d_model.trainable = True

            r_loss,_ = d_model.train_on_batch(x_real,y_real)
            f_loss,_ = d_model.train_on_batch(x_fake,y_fake)

            d_loss = 0.5*(r_loss+f_loss)

            d_model.trainable = False

            x_rv = generate_rv(rv_dim,batch_size)
            y = np.ones(shape=(x_rv.shape[0],1))

            g_loss,_ = gan_model.train_on_batch(x_rv,y)

        print(f'Epoch: {e+1}, d_loss: {d_loss}, g_loss: {g_loss}')
        if e%10 == 0:
            save_fig(g_model,rv_dim,e)

keras.utils.disable_interactive_logging()
rv_dim = 50
g_model = generator(rv_dim)
d_model = discriminator()
gan = gan_model(g_model,d_model)
data = load_real_data()
train(data,g_model,d_model,gan,rv_dim)
keras.utils.enable_interactive_logging()
