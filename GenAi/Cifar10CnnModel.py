# Import necessary libraries
import tensorflow as tf
from tensorflow import keras  # For building the model using Sequential and Functional API
from keras import layers       # For using different types of layers
from keras.datasets import cifar10  # Dataset of 60,000 32x32 colour images in 10 classes
from keras.datasets import fashion_mnist  # Not used here, just imported
import numpy as np
import matplotlib.pyplot as plt  # For visualizing images and graphs

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Display shapes of training data
print(x_train.shape)  # Expected output: (50000, 32, 32, 3)
print(y_train.shape)  # Expected output: (50000, 1)

# Print the pixel values of the first training image
print(x_train[0])

# Plot the first 25 images in the training set
plt.figure(figsize=(10, 10))
for i in range(1, 26):
    plt.subplot(5, 5, i)  # Create a 5x5 grid of subplots
    plt.imshow(x_train[i])  # Show the image
    plt.title(y_train[i])   # Display the label as title
    plt.axis('off')         # Hide axis ticks
plt.show()

# These lines are used for grayscale datasets like MNIST/Fashion-MNIST, not needed here
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# Confirm image shapes again
print(x_train.shape)
print(x_test.shape)

# Set input shape and number of classes
input_shape = (32, 32, 3)
num_classes = 10

# Build the CNN model using Sequential API
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),            # Input layer
        keras.layers.Rescaling(1. / 255),          # Normalize pixel values to [0, 1]
        
        # First Convolutional + MaxPooling block
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),
        
        # Second Convolutional + MaxPooling block
        keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Third and fourth Conv layers (no pooling between them)
        keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Flatten the output from 3D to 1D
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),  # Dropout to prevent overfitting

        # Dense (fully connected) layers
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),  # Another dropout layer

        # Output layer with softmax for multi-class classification
        keras.layers.Dense(num_classes, activation='softmax')
    ]
)

# Display the model's architecture
model.summary()

# Compile the model with optimizer, loss function and evaluation metric
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # Adam optimizer with low learning rate
    loss='sparse_categorical_crossentropy',               # Suitable for integer-labeled targets
    metrics=['accuracy']                                  # Track accuracy during training
)

# Train the model with training data
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1  # 10% of training data used for validation
)

# Evaluate the trained model on test data
model.evaluate(x_test, y_test)

# Access training history
history.history.keys()  # Shows 'accuracy', 'val_accuracy', 'loss', 'val_loss'

# Extract accuracy and loss for plotting
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
