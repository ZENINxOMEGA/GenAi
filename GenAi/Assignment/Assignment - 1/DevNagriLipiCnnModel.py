# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set dataset paths
train_path = "GenAi/Assignment/Devnagri Lipi/Hindi/Train"
test_path = "GenAi/Assignment/Devnagri Lipi/Hindi/Test"

# Load Devanagari dataset from directories
batch_size = 32
img_size = (32, 32)

train_ds = keras.preprocessing.image_dataset_from_directory(
    train_path,
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_path,
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Get class names and number of classes before prefetch
class_names = train_ds.class_names
num_classes = len(class_names)

# Visualize the first 25 images from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy())
        plt.axis("off")
plt.show()

# Prefetch datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Set input shape
input_shape = (32, 32, 3)

# Build the CNN model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Rescaling(1. / 255),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ]
)

# Display the model's architecture
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds
)

# Evaluate the model
model.evaluate(test_ds)

# Extract training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation accuracy/loss
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()
