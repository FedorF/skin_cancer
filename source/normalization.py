import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
import shutil
import os
import cv2 as cv
from glob import glob
import skimage.measure
from source.models import cNN
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

PATH_TO_KERAS_MODEL = './models/keras_model'
PATH_TO_KERAS_MODEL_WEIGHTS = './models/keras_model_weights.h5'

PATH_TO_DATA = './data/train'
neg_dir = glob(os.path.join(PATH_TO_DATA, 'negative/*.jpg'))
pos_dir = glob(os.path.join(PATH_TO_DATA, 'positive/*.jpg'))



def normalize(X):
    return ((X - np.mean(X, axis=0)) / np.std(X, axis=0))

def crop(img):
    h, w, c = img.shape
    if h > w:
        delta = (h - w) // 2
        img = img[delta:h-delta, :, :]
    elif h < w:
        delta = (w - h) // 2
        img = img[:, delta:w-delta, :]

    return img

N = len(neg_dir) + len(pos_dir)

H, W, C = 64, 64, 3
batch_size = 75

X = np.zeros((N, H, W, C))
y = np.zeros(N)

i = 0
for img_path in neg_dir:
    if i % 100 == 0:
        print(i)

    img = cv.imread(img_path)
    img = cv.resize(crop(img), (H, W))
    X[i, :, :, :] = img
    i += 1

for img_path in pos_dir:
    if i % 100 == 0:
        print(i)
    img = cv.imread(img_path)
    img = cv.resize(crop(img), (H, W))
    X[i, :, :, :] = img
    y[i] = 1
    i += 1
print(X.shape)

X = normalize(X)

def cNN(input_shape, droupout=True, n_classes=2):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if droupout:
        model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if droupout:
        model.add(Dropout(.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if droupout:
        model.add(Dropout(.25))

    model.add(Flatten())
    model.add((Dense(512, activation='relu')))
    if droupout:
        model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    return model

model = cNN((H, W, C))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, batch_size=batch_size, validation_split=0.2, verbose=1)

# Save model
model_json = model.to_json()
with open(PATH_TO_KERAS_MODEL, 'w') as f:
    f.write(model_json)

model.save_weights(PATH_TO_KERAS_MODEL_WEIGHTS)