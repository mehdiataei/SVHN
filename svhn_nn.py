# -*- coding: utf-8 -*-
"""
Spyder Editor

Python 3

"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np
import scipy.io as io

#Importing data
train_set = io.loadmat("train_32x32.mat")
test_set = io.loadmat("test_32x32.mat")

x_train = train_set['X']
y_train = train_set['y']
x_test = test_set['X']
y_test = test_set['y']

# Input shape
rows, cols = x_test.shape[1], x_test.shape[2]

# Reshaping test and training data sets
x_train = np.moveaxis(x_train,[3,0,1,2],[0,1,2,3])
x_train = x_train.astype('float32')
x_test = np.moveaxis(x_test,[3,0,1,2],[0,1,2,3])
x_test = np.asarray(x_test)
x_test = x_test.astype('float32')

# Input normalization
x_train /= x_train.max()
x_test /= x_test.max()

# One-hot encoding
y_test = [y-1 for y in y_test]
y_train = [y-1 for y in y_train]
classes = 10
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)

# Defining data type for memory optimization
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Featurewise normalization
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=False,
    zca_whitening=False, 
    zca_epsilon=1e-06)

datagen.fit(x_train)
    
def schn(x_train, y_train, x_test, y_test):
    model = Sequential()
    classes = 10
    batch_size = 128
    epochs = 40
    
    # Convolutional deep neural network model development
    model.add(Conv2D(32, 3, 3, border_mode='same',
                            input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(BatchNormalization(weights=None))
    model.add(Activation('relu'))
    model.add(BatchNormalization(weights=None))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    # Compilation
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])

    # Fit the model on the batches
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=x_train.shape[0],
                        nb_epoch=epochs)

    # Model evaluation
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)
   
schn(x_train, y_train, x_test, y_test)


"""
572/572 [==============================] - 55s - loss: 0.1408 - acc: 0.9535

The test score is [0.19329260257175392, 0.95234345181037275]

"""