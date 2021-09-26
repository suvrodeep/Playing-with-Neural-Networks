#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
import pandas as pd
import numpy as np
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Check dimensions of image
x_train[0].shape

# View a sample image
# plt.imshow(x_train[10])

x_train
x_train[10].max()

y_train

print("Train data shape:{}".format(x_train.shape))
print("Test data shape:{}".format(x_test.shape))

print("Number of classes training labels:{}".format(len(pd.unique(y_train))))
print("Unique values in training labels:{}".format(pd.unique(y_train)))

# Normalize image pixel values
x_train = x_train / 255
x_test = x_test / 255

# Prepare one-hot encoded y-values
y_train_cat = keras.utils.to_categorical(y=y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y=y_test, num_classes=10)

# Reshaping data to batch_size, width, height, color_channel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# Prepare GPU
def prep_devices():
    phy_gpus = tf.config.list_physical_devices(device_type='GPU')

    for gpu in phy_gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    log_gpus = tf.config.list_logical_devices(device_type='GPU')
    phy_cpus = tf.config.list_physical_devices(device_type='CPU')
    log_cpus = tf.config.list_logical_devices(device_type='CPU')

    print("Physical GPUs: {}\tLogical GPUs: {}\nPhysical CPUs: {}\tLogical CPUs: {}".format(len(phy_gpus),
                                                                                            len(log_gpus),
                                                                                            len(phy_cpus),
                                                                                            len(log_cpus)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
prep_devices()


# Build model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

# Create early stopping callback
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

# Fit model
backend.clear_session()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=x_train, y=y_train_cat, epochs=50, validation_split=0.1, callbacks=[early_stop])


# Model evaluation
metrics = pd.DataFrame(model.history.history)
metrics


# Evaluate graphically
metrics = pd.DataFrame(model.history.history)
# sns.lineplot(data=metrics[['accuracy', 'val_accuracy']])


# Compute predicted classes and model performance metrics
y_pred = np.argmax(model.predict(x_test), axis=-1)

print(classification_report(y_true=y_test, y_pred=y_pred))
print("\n")
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
