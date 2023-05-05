# 1. 通过使用深度学习框架来简洁的实现 线性回归模型 生成数据集
import numpy as np
import random
import torch
from d2l import torch as d2l

#  根据带有噪声的线性模型构造一个人造数据集。 我们使用线性模型参数w=[2,−3.4]^⊤、b=4.2和噪声项ϵ生成数据集及其标签：y=Xw+b+ϵ
def synthetic_data(w, b, num_examples):  # num_examples:n个样本
    '''生成 y=Xw+b+噪声'''
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成 X，他是一个均值为0，方差为1的随机数，他的大小: 行为num_examples，列为w的长度表示多少个feature
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 加入一些噪音，均值为0 ，方差为0.01，形状和y是一样
    return X, y.reshape((-1, 1))  # 把X和y作为一个列向量返回,-1代表自动计算维度


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # synthetic_data这个函数返回的是特征和标签，相当于分别吧真实的房屋‘关键因素’和对应的‘房价’列出来了
# print(features,labels)
# 这里的X指房屋的关键因素集，长度len（w）即列数，表明有len(w)个关键因素，这里是2，比如‘卧室个数’和‘住房面积’两个关键因素，X的行数num_example=房屋的数量
# 以上相当于去市场调研收集真实的房屋数据

# #输出显示
print('features:', features[0], '\nlabel:', labels[0])

# 可视化

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(),  # detach()分离出数值，不再含有梯度
                labels.detach().numpy(), 1)       # scatter()函数的最后一个1是绘制点直径大小

# 把feature的第一列和labels绘出来，是有线性相关的性质
# 显示结果如下图：

d2l.set_figsize()
d2l.plt.scatter(features[:, 0].detach().numpy(),
                labels.detach().numpy(), 10)

# 修改为绘制第所有feature的第0列和label的关系，点的大小设置为10
# 第0列正相关，第1列负相关   w=[2,−3.4]⊤   0列对应w的2，斜率为正的直线；1列对应w的-3.4,斜率为负的直线，所以一个正相关一个负相关
# 显示结果如下图：


# 2. 每次读取一个小批量：定义一个data_iter 函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量

def data_iter(batch_size, features, labels):  # data_iter函数接收批量大小、特征矩阵和标签向量作为输入
    num_examples = len(features)
    indices = list(range(num_examples))  # 生成每个样本的index，随机读取，没有特定顺序。range随机生成0 —（n-1）,然后转化成python的list
    random.shuffle(indices)  # 将下标全都打乱，打乱之后就可以随机的顺序去访问一个样本
    for i in range(0, num_examples, batch_size):  # 每次从0开始到num_examples，每次跳batch_size个大小
        batch_indices = torch.tensor(  # 把batch_size的index找出来，因为可能会超出我们的样本个数，所以最后如果没有拿满的话，会取出最小值，所以使用min
            indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
# 只是indices这个List被打乱了，feature和labels都是顺序的，用循环才能随机的放进去
# （构造一个随机样本。把样本的顺序打乱，然后间隔相同访问，也能达到随机的目的）
# yield是构造一个生成器，返回迭代器。yield就是return返回一个值，并且记住返回的位置，下次迭代就从这个开始

batch_size = 10

for X, y in data_iter(batch_size, features, labels):  # 调用data_iter这个函数返回iterator（迭代器），从中拿到X和y
    print(X, '\n', y)  # 给我一些样本标号每次随机的从里面选取一个样本返回出来参与计算
    break



# 3. 定义 初始化模型参数

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # w:size为2行1列,随机初始化成均值为0，方差为0.01的正态分布，requires=true是指需要计算梯度
b = torch.zeros(1, requires_grad=True)  # 对于偏差来说直接为0，1表示为一个标量，因为也需要进行更新所以为True

# 广播机制：当我们用一个向量加一个标量时，标量会被加到每一个分量上


# 4. 定义模型

def linreg(X, w, b):
	    """线性回归模型。"""
	    return torch.matmul(X, w) + b    #矩阵乘以向量再加上偏差

# 5. 定义损失函数

def squared_loss(y_hat, y):         #y_hat是预测值，y是真实值
	    """均方损失。"""
	    return (y_hat - y.reshape(y_hat.shape))**2 / 2      #按元素做减法，按元素做平方，再除以2  （这里没有做均方）

# 6. 定义优化算法

def sgd(params, lr, batch_size):  # 优化算法是sgd，他的输入是：params给定所有的参数,这个是一个list包含了w和b，lr是学习率，和batch_size大小
    """小批量随机梯度下降。"""
    with torch.no_grad():  # 这里更新的时候不需要参与梯度计算所以是no_grad
        for param in params:  # 对于参数中的每一个参数，可能是w可能是b
            param -= lr * param.grad / batch_size  # 参数减去learning rate乘以他的梯度（梯度会存在.grad中）。上面的损失函数中没有求均值，所以这里除以了batch_size求均值，因为乘法对于梯度是一个线性的关系，所以除以在上面损失函数那里定义和这里是一样的效果
            print(f'参数为: {param}')
            param.grad.zero_()  # 把梯度设置为0，因为pytorch不会自动的设置梯度为0，需要手动，下次计算梯度的时候就不会与这次相关了
            print(f'清零后的梯度: {param.grad}')
# 我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size）来归一化步长，这样步长大小就不会取决于我们对批量大小的选择
# 更新数据时不需要求导
# pytorch 会不断累加变量的梯度，所以每更新一次参数，就要让其对应的梯度清零

# 7. 训练过程
lr = 0.03  # 首先指定一些超参数：学习率为0.03
num_epochs = 3  # epoch为3表示把整个数据扫3遍
net = linreg  # network为linreg前面定义的线性回归模型
loss = squared_loss  # loss为均方损失

for epoch in range(num_epochs):  # 训练的过程基本是两层for循环（loop）,第一次for循环是对数据扫一遍
    print(f'第: {epoch}轮')
    i = 1
    for X, y in data_iter(batch_size, features, labels):  # 对于每一次拿出一个批量大小的X和y
        print(f'第: {i}次')
        i += 1
        l = loss(net(X, w, b), y)  # 把X,w,b放进network中进行预测，把预测的y和真实的y来做损失，则损失就是一个长为批量大小的一个向量，是X和y的小批量损失
        # l(loss)的形状是（'batch_size',1）,而不是一个标量
        print(l)
        l.sum().backward()  # 对loss求和然后算梯度。计算关于['w','b']的梯度
        print(f'w的梯度为: {w.grad}')
        print(f'b的梯度为: {b.grad}')
        sgd([w, b], lr, batch_size)  # 算完梯度之后就可以访问梯度了，使用sgd对w和b进行更新。使用参数的梯度对参数进行更新
    # 对数据扫完一遍之后来评价一下进度，这块是不需要计算梯度的，所以放在no_grad里面
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)  # 把整个features，整个数据传进去计算他的预测和真实的labels做一下损失，然后print
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 求和本身是让l(即loss)以标量的形式表现出来（通过sum()转化标量之后在求梯度）（一般都是对一个标量进行求导，所以我们先对y进行求和再求导：见前面自动求导笔记）。
# 不求和是向量，梯度算下来就是变成矩阵了，形状没有办法对应
# 求梯度是对于l中每一个分量都是单独求的，l（loss）是一个向量每个元素表示一个样本的误差除以批量数
# 如果没有no_grad，在评估模型的时候也进行梯度优化过程了


# 8. 比较真实参数和通过训练学到的参数来评估训练的成功程度

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# 9. 不同超参数的选择会有什么样的不同的效果（修改超参数）

# lr = 0.001  # 学习率很小的时候,发现损失很大，即使把epoch调大，loss仍然很大
# num_epochs = 3
# net = linreg
# loss = squared_loss
#
# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd([w, b], lr, batch_size)
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
#
# # 重新运行的时候要把上面w和b初始化的代码重新运行一下，重新初始化w和b





