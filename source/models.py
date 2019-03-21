from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import torch
from torch import nn

def cNN(input_shape, droupout, n_classes):
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


class CNN(nn.Module):

    def __init__(self, H, W, reg=0):

        super(CNN, self).__init__()
        self.reg = reg

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.out = nn.Sequential(
            nn.Linear(H // 4 * W // 4 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def l1_regularization(self):
        s = 0
        for w in self.parameters():
            s += torch.abs(w).sum()
        return s

    def l2_regularization(self):
        s = 0
        for w in self.parameters():
            s += torch.sum(w * w)
        return s

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view((x.shape[0], -1))
        output = self.out(x)
        return output