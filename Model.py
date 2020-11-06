# -*- coding: utf-8 -*-


from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.losses import categorical_crossentropy


class LeNet(Sequential):
    """
    LeNet-5 model
    http://datahacker.rs/deep-learning-lenet-5-architecture/
    """
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        # padding='same'，論文輸入為32*32，因為mnist為28*28，所以第一層加上padding='same'
        # C1
        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape,
                        padding='same'))
        # S2
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 這層要留意
        # C3
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # S4
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # 第一種方式
        # self.add(Flatten())
        # self.add(Dense(120, activation='tanh'))

        # 根據論文的第二種方式，input shape: 5*5，經過 filter後為1*1
        # C5
        self.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(Flatten())

        # FC6
        self.add(Dense(84, activation='tanh'))

        # Output
        self.add(Dense(nb_classes, activation='softmax'))

        # compile
        self.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])