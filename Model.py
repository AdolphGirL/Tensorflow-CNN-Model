# -*- coding: utf-8 -*-


from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import Layer
import tensorflow as tf


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


class LRN(Layer):
    def __init__(self):
        super(LRN, self).__init__()
        self.depth_radius = 2
        self.bias = 1
        self.alpha = 1e-4
        self.beta = 0.75

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius,
                         bias=self.bias, alpha=self.alpha, beta=self.beta)


class AlexNet(Sequential):
    """
    AlexNet
    https://blog.csdn.net/weixin_42201701/article/details/98098281
    """

    def __init__(self, input_shape, nb_classes):
        super().__init__()

        # input_shape 227*227*3
        # 卷積層-1(兩個gpu同時訓練)，卷積核11*11，filter48，步長4，激活函數relu，輸出(227-11)/4 + 1 = 55 -> (55*55*96)
        # padding='same' ??
        self.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape,
                        padding='same'))
        # 卷積層-1-池化，pool_size=(3,3) strides=(2, 2)，輸出(55-3)/2 + 1 = 27 -> (27*27*96)
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 局部響應歸一化
        self.add(LRN())

        # 卷積層-2(兩個gpu同時訓練)，卷積核5*5，filter128，步長1，激活函數relu，padding='same'，輸出 (27*27*256)
        self.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
        # 卷積層-2-池化，pool_size=(3,3) strides=(2, 2)，輸出(27-3)/2 + 1 = 13 -> (13*13*256)
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 局部響應歸一化
        self.add(LRN())

        # 卷積層-3(兩個gpu同時訓練)，卷積核3*3，filter192，步長1，激活函數relu，padding='same'，輸出 (13*13*384)
        self.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

        # 卷積層-4(兩個gpu同時訓練)，卷積核3*3，filter192，步長1，激活函數relu，padding='same'，輸出 (13*13*384)
        self.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

        # 卷積層-5(兩個gpu同時訓練)，卷積核3*3，filter128，步長1，激活函數relu，padding='same'，輸出 (13*13*256)
        self.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
        # 卷積層-5-池化，pool_size=(3,3) strides=(2, 2)，輸出(13-3)/2 + 1 = 6 -> (6*6*256)
        self.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 攤平
        self.add(Flatten())

        # 測試使用5分類，所以改一下
        """
        # 全連接-6(兩個gpu同時訓練) 2048個神經元，激活函數relu
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        
        # 全連接-7(兩個gpu同時訓練) 2048個神經元，激活函數relu
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        """

        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))

        # 全連接-7(兩個gpu同時訓練) 2048個神經元，激活函數relu
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))

        # 全連接-8 1000個神經元，激活函數softmax
        self.add(Dense(nb_classes, activation='softmax'))


        # compile
        self.compile(optimizer='sgd', loss=sparse_categorical_crossentropy, metrics=['accuracy'])
