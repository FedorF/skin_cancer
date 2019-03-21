import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from source.models import cNN
from glob import glob
import os
import cv2 as cv
import numpy as np

PATH_DATA = './data/images/'
PATH_METADATA = './data/HAM10000_metadata.csv'
PATH_AUGMENTATION = './data/augmentation/'
metadata = pd.read_csv(PATH_METADATA)
labels = {k: v for v, k in enumerate(metadata.dx.unique())}
metadata['label'] = metadata['dx'].map(labels)

imgs = os.listdir(PATH_DATA)


def normalize(X):
    return ((X - np.mean(X, axis=0)) / np.std(X, axis=0))


N, H, W, C = len(imgs), 450, 600, 3
X = []
y = np.zeros(N)
i = 0

for img_path in imgs:
    #img = cv.imread(PATH_DATA + img_path)
    #X.append(img)
    #print(type(img_path.split('.')[0]))
    y[i] = metadata[metadata['image_id'] == img_path.split('.')[0]]['label'].values[0]
    i += 1

dataset = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=180,
                             width_shift_range=.1,
                             height_shift_range=.1,
                             zoom_range=.1,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

augmentation = dataset.flow_from_directory(PATH_DATA,
                                           save_to_dir=PATH_AUGMENTATION,
                                           save_format='jpg',
                                           target_size=(256, 256),
                                           batch_size=50,
                                           classes=y)

input_size = (H, W, C)
droupout = .4
n_classes = 7
model = cNN(input_size, droupout, n_classes)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
