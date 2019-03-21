from keras.models import model_from_json
from glob import glob
import os
import numpy as np
import cv2 as cv


PATH_TO_KERAS_MODEL = './models/keras_model'
PATH_TO_KERAS_MODEL_WEIGHTS = './models/keras_model_weights.h5'

PATH_TO_DATA = './data/test'
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

X_test = np.zeros((N, H, W, C))
y_test = np.zeros(N)

i = 0
for img_path in neg_dir:
    if i % 100 == 0:
        print(i)

    img = cv.imread(img_path)
    img = cv.resize(crop(img), (H, W))
    X_test[i, :, :, :] = img
    i += 1

for img_path in pos_dir:
    if i % 100 == 0:
        print(i)
    img = cv.imread(img_path)
    img = cv.resize(crop(img), (H, W))
    X_test[i, :, :, :] = img
    y_test[i] = 1
    i += 1
print(X_test.shape)

X_test = normalize(X_test)



with open(PATH_TO_KERAS_MODEL, 'r') as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights(PATH_TO_KERAS_MODEL_WEIGHTS)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
scores = model.evaluate(X_test, y_test, verbose=1)