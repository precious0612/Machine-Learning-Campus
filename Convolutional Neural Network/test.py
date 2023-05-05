import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras

from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from matplotlib import pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test1 = X_test
Y_test1 = Y_test
# 处理图像特征
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# 处理标签
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
# print(X_train)
# print("-----------")
# print(Y_train)

# 搭建AlexNet网络模型
# 建立第一层卷积
model = Sequential()
model.add(Conv2D(
    filters=96,
    kernel_size=(11, 11),
    strides=4,
    padding="same",
    input_shape=(28, 28, 1),
    activation="relu"
))

# 搭建BN层
model.add(BatchNormalization())
# 搭建第一层重叠最大池化层
model.add(MaxPool2D(
    pool_size=(3, 3),
    strides=2,
    padding="same"
))

# 建立第二层卷积
model.add(Conv2D(
    filters=256,
    kernel_size=(5, 5),
    strides=1,
    padding="same",
    activation="relu"
))
# 搭建BN层
model.add(BatchNormalization())
# 搭建第二层池化层
model.add(MaxPool2D(
    pool_size=(3, 3),
    strides=2,
    padding="same",
))

# 搭建第三层卷积
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    strides=1,
    padding="same",
    activation="relu",
))

# 搭建第四层卷积
model.add(Conv2D(
    filters=384,
    kernel_size=(3, 3),
    strides=1,
    padding="same",
    activation="relu"
))

# 搭建第五卷积层
model.add(Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=1,
    padding='same',
    activation="relu"
))
model.add(MaxPool2D(
    pool_size=(3, 3),
    strides=2,
    padding="same"
))

# 搭建第六层：全连接层
# 在搭建全连接层之前，必须使用Flatten()降维
model.add(Flatten())
# 全连接层
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
# 搭建第七层：全连接层
model.add(Dense(2048, activation="relu"))
model.add(Dropout(0.5))
# 搭建第八层：全连接层即输出层
model.add(Dense(10, activation="softmax"))
model.summary()

# 编译
model.compile(
    loss="categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

# 训练
n_epoch = 10
n_batch = 128
training = model.fit(
    X_train,
    Y_train,
    epochs=n_epoch,
    batch_size=n_batch,
    verbose=1,
    validation_split=0.20
)


# 画出准确率随着epoch的变化图
def show_train(tr, train, validation):
    plt.plot(training.history[train], linestyle="-", color="b")
    plt.plot(training.history[validation], linestyle="--", color="r")
    plt.title("trianing_history")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="lower right")
    plt.show()


show_train(training, "accuracy", "val_accuracy")


# 画出误差随着epoch的变化图
def show_train(tr, train, validation):
    plt.plot(training.history[train], linestyle="-", color="b")
    plt.plot(training.history[validation], linestyle="--", color="r")
    plt.title("trianing_history")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()


show_train(training, "loss", "val_loss")

# 评估
test = model.evaluate(X_train, Y_train, verbose=1)
print("误差：", test[0])
print("准确率：", test[1])


# 预测
def image_show(image):  # 画图
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap="binary")
    plt.show()


prediction = model.predict_classes(X_test)


def pre_result(i):
    image_show(X_test1[i])
    print("Y-test:", Y_test1[i])
    print("预测值：", prediction[i])


pre_result(0)
pre_result(1)