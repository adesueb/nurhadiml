import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DATADIR = "dataset"
TESTDIR = "test"
LABELS = ["indosiar", "indosiar_iklan", "sctv", "sctv_iklan"]

X_TRAIN = []
Y_TRAIN = []

IMG_SIZE = 100
for category in LABELS:
    path = os.path.join(DATADIR, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_TRAIN.append(new_array)
            Y_TRAIN.append(class_num)
        except Exception as e:
            pass
            
X_TRAIN = np.array(X_TRAIN).reshape(-1, IMG_SIZE, IMG_SIZE,3)
Y_TRAIN = np.array(Y_TRAIN)

X_TRAIN = X_TRAIN/255

model = Sequential()
model.add(Conv2D(32, (5,5), input_shape = X_TRAIN.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_TRAIN,Y_TRAIN, epochs=10, validation_split=0.1)

#for testing
for img in os.listdir(TESTDIR):
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        new_img = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_shape = new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        predictions = model.predict(new_shape)
        plt.imshow(new_img)
        print(predictions)
        print(LABELS[np.argmax(predictions)])
    except Exception as e:
        pass


