# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from Model import AlexNet
from tensorflow.keras.utils import to_categorical
import datetime


def load_pokemon():
    """
    0: 妙蛙種子, 1:小火龍, 2:超夢, 3:皮卡丘, 4:傑尼龜
    :return: train_x, train_y, test
    """
    file_dir = os.pardir + os.sep + 'data-file' + os.sep + 'youthai-competition'
    np_train_file = os.path.join(file_dir, 'pokemon_train.npy')
    np_test_file = os.path.join(file_dir, 'pokemon_test.npy')

    train = np.load(np_train_file)
    # train資料，只有圖與label的資料 -> test.shape=(1000, 49153)，train_x[i].reshape(128, 128, 3)
    # print(train.shape)
    train_x, train_y = train[:, 1:], train[:, 0]
    train_x = train_x.reshape(-1, 128, 128, 3)

    # test資料，只有圖的資料 -> test.shape=(167, 49152)
    test = np.load(np_test_file)
    test = test.reshape(-1, 128, 128, 3)
    # print(test.shape)
    return train_x, train_y, test


if __name__ == '__main__':
    num_classes = 5
    logger_name = '[AlexNet.py]'

    train_images, train_labels, y = load_pokemon()
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    print(logger_name + ' train_images shape: {}'.format(train_images.shape))
    print(logger_name + ' train_images image shape: {}'.format(train_images[0].shape))
    print(logger_name + ' train_labels shape: {}'.format(train_labels.shape))
    print(logger_name + ' y image shape: {}'.format(y.shape))

    # Data normalization
    train_images = train_images.astype('float32')
    y = y.astype('float32')

    # train_images /= 255
    # y /= 255

    model = AlexNet(train_images[0].shape, num_classes)
    model.summary()

    # log_dir = "logs/alextnetfit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_images, y=train_labels, verbose=0)

    prediction_values = model.predict_classes(y)

    # 看一下預測結果
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(50):
        # xticks yticks設定為空
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(y[i, :], cmap=plt.cm.gray_r, interpolation='nearest')

        # y_test[i] shape = (10, )，即為一個向量，非一為矩陣
        ax.text(0, 7, prediction_values[i], color='red')

    plt.show()
