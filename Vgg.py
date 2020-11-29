# -*- coding: utf-8 -*-


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Layer
import tensorflow as tf


class VGG16(Sequential):
    """
    13層卷積層
    3層全連接層
    5層最大池化層

    one-hot encoding
    """
    def __init__(self, input_shape, nb_classes):
        super().__init__()

        self.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv1a',
                        input_shape=input_shape))
        # 另一種寫法
        self.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1b'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1'))

        self.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2a'))
        self.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2b'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2'))

        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3a'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3b'))
        self.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3c'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3'))

        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4a'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4b'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4c'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4'))

        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5a'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5b'))
        self.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5c'))
        self.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool5'))

        # 攤平
        self.add(Flatten())

        self.add(Dense(4096, activation='relu', name='fc6'))
        self.add(Dense(4096, activation='relu', name='fc7'))
        self.add(Dense(nb_classes, activation='softmax', name='fc8'))

        # compile
        self.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])


if __name__ == '__main__':
    vgg = VGG16((32, 32, 3), 10)
    vgg.summary()

