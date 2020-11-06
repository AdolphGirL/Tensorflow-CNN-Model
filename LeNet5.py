# -*- coding: utf-8 -*-


import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from Model import LeNet


logger_name = '[LeNet5.py]'
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(logger_name + ' x_train shape: {}'.format(x_train.shape))
print(logger_name + ' x_test samples: {}'.format(x_test.shape))
print(logger_name + ' x_train image shape: {}'.format(x_train[0].shape))

# Add a new axis，將灰階陣列改為3通道的image格式(灰階只會有一通道)
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
print(logger_name + ' x_train shape: {}'.format(x_train.shape))
print(logger_name + ' x_test samples: {}'.format(x_test.shape))
print(logger_name + ' x_train image shape: {}'.format(x_train[0].shape))

# Convert class vectors to binary class matrices.
# mnist資料及輸出為10類
num_classes = 10
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(logger_name + ' y_train shape: {}'.format(y_train.shape))
print(logger_name + ' y_test samples: {}'.format(y_test.shape))
print(logger_name + ' y_test label shape: {}'.format(y_test[0].shape))

# Data normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = LeNet(x_train[0].shape, num_classes)
model.summary()

# TensorBoard
# Place the logs in a timestamped subdirectory
# This allows to easy select different training runs
# In order not to overwrite some data,
# it is useful to have a name with a timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# tf.keras.callback.TensorBoard ensures that logs are created and stored
# We need to pass callback object to the fit method
# The way to do this is by passing the list of callback objects, which is in our case just one
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 看一下訓練結果
model.fit(x_train, y=y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback], verbose=0)
# %tensorboard --logdir logs/fit

prediction_values = model.predict_classes(x_test)

# 看一下預測結果
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(50):
    # xticks yticks設定為空
    ax = fig.add_subplot(5, 10, i+1, xticks=[], yticks=[])
    ax.imshow(x_test[i, :].reshape((28, 28)),
              cmap=plt.cm.gray_r, interpolation='nearest')

    # y_test[i] shape = (10, )，即為一個向量，非一為矩陣
    if prediction_values[i] == np.argmax(y_test[i]):
        ax.text(0, 7, prediction_values[i], color='blue')
    else:
        ax.text(0, 7, prediction_values[i], color='red')

plt.show()
