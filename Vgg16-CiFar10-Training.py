# -*- coding: utf-8 -*-


from model.Vgg import VGG16
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical


logger_name = '[Vgg16-CiFar10-Training.py]: '
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(logger_name + 'x_train shape: {}'.format(x_train.shape))
print(logger_name + 'y_train shape: {}'.format(y_train.shape))
print(logger_name + 'x_test shape: {}'.format(x_test.shape))
print(logger_name + 'y_test shape: {}'.format(y_test.shape))

# 看一下原本的圖片

# fig = plt.figure(figsize=(15, 7))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
#
# for i in range(50):
#     # xticks yticks設定為空
#     ax = fig.add_subplot(5, 10, i+1, xticks=[], yticks=[])
#     ax.imshow(x_test[i, :, :, :], cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.text(0, 7, y_train[i], color='red')
#
# plt.show()

# one-hot encoding
num_classes = 10
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

model = VGG16(input_shape=(32, 32, 3), nb_classes=num_classes)
model.summary()

# TensorBoard
log_dir = "logs/vgg16/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 加入順練時的callback function
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 電腦配備問題，epochs沒設太高
# verbose=2，one line per epoch
model.fit(x_train, y=y_train, epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback], verbose=2)


prediction_values = model.predict_classes(x_test)

# 看一下預測結果
fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(50):
    ax = fig.add_subplot(5, 10, i+1, xticks=[], yticks=[])
    ax.imshow(x_test[i, :, :, :], cmap=plt.cm.gray_r, interpolation='nearest')

    # y_test[i] 已經過one-hot coding，所以取最大值的位置
    if prediction_values[i] == np.argmax(y_test[i]):
        ax.text(0, 7, prediction_values[i], color='blue')
    else:
        ax.text(0, 7, prediction_values[i], color='red')

plt.show()
