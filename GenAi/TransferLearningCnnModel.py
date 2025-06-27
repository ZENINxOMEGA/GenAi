import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)

plt.figure(figsize = (10,10))
for i in range(1,26):
    plt.subplot(5,5,i)
    plt.imshow(x_train[i])
    plt.title(y_train[i])
    plt.axis('off')
plt.show()

plt.imshow(x_train[9])
plt.show()

img = image.img_to_array(tf.image.resize(x_train[9], [224, 224],method=tf.image.ResizeMethod.BILINEAR))
plt.imshow(img.astype('int32'))
plt.show()

model = ResNet50(weights='imagenet')
x = np.expand_dims(img, axis=0)# x = preprocess_input(x)
preds = model.predict(x)

print("Predicted",decode_predictions(preds, top=5))

model = ResNet50(include_top=False,input_shape=(224,224,3))
model.summary()

for layer in model.layers:
    layer.trainable = False
x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024,activation = 'relu')(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128,activation = 'relu')(x)
x = keras.layers.Dropout(0.3)(x)
output = keras.layers.Dense(10,activation='softmax')(x)
TLModel = keras.Model(inputs = model.input,outputs = output)
TLModel.summary()

# x_train = tf.image.resize(x_train/255.0,(224,224))
# x_test = tf.image.resize(x_test/255.0,(224,224))
np.unique(y_train[:1000],return_counts=True)

nx_train = x_train[:5000]
ny_train = y_train[:5000]

nx_test = x_test[:1000]
ny_test = y_test[:1000]
nx_train = tf.image.resize(nx_train,(224,224))
nx_test = tf.image.resize(nx_test,(224,224))

nx_train = preprocess_input(nx_train)
nx_test = preprocess_input(nx_test)
TLModel.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
TLModel.fit(nx_train,ny_train,batch_size=32,epochs=20,validation_data=(nx_test,ny_test))