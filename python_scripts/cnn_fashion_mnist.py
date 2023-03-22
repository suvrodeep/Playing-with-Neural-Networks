import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras import backend
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Check dimensions of image
print(x_train[0].shape)

# View a sample image
plt.imshow(x_train[10])

##################################################
# Explore training data and labels
##################################################

print(x_train)

print(x_train[10].max())

print(y_train)

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

    # Disabled for Nvidia GPU 3060 Ti
    # for gpu in phy_gpus:
    #     tf.config.experimental.set_memory_growth(device=gpu, enable=True)

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
def model_func(clear_session=True):
    if clear_session:
        backend.clear_session()

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

    return model


# Define function to clear previous run model
def train_model(optimizer="adam", epochs=100, metrics=None, patience=5, val_split=0.1, batch_size=500):
    # Fit model
    if metrics is None:
        metrics = ["accuracy"]
    model = model_func()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)
    model.fit(x=x_train, y=y_train_cat, epochs=epochs, validation_split=val_split, batch_size=batch_size,
              callbacks=[early_stop])

    return model


trained_model = train_model()

# Model evaluation
model_metrics = pd.DataFrame(trained_model.history.history)
sns.lineplot(data=model_metrics[['accuracy', 'val_accuracy']])

# Compute predicted classes and model performance metrics
y_pred = np.argmax(trained_model.predict(x_test), axis=-1)

print(classification_report(y_true=y_test, y_pred=y_pred))
print("\n")
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

