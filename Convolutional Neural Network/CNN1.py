"""
本程序有两个功能：
1、定义卷积操作
2、定义卷积层

"""

import torch
from torch import nn
from d2l import torch as d2l

# 定义卷积操作函数，X是输入(Nh*Nw)，K为核矩阵(Kh,Kw)
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape  # 核矩阵的行和列
    # 初始化输出的维度，公式(Nh-Kh+1)*(Nw-Kw+1)
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 验证PPT上的卷积操作，X为输入矩阵，K为卷积核
X1 = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
Y1 = corr2d(X1, K)
print(Y1)

# 定义卷积层，实现前向卷积操作。初始化W和b的值，同时定义计算卷积的输出Y = W * X + b
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

"""

验证边缘检测的效果，通过找到像素变化的位置，来检测图像中不同颜色的边缘。 
首先，我们构造一个像素的黑白图像。中间四列为黑色（0），其余像素为白色（1）
"""
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

# 接下来，我们构造一个高度为、宽度为的卷积核K。当进行互相关运算时，
# 如果水平相邻的两元素相同，则输出为零，否则输出为非零。
K = torch.tensor([[1.0, -1.0]])

# 对参数X（输入）和K（卷积核）执行互相关运算。
# 如下所示，输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为。
Y2 = corr2d(X, K)
print(Y2)

# 刚才的核矩阵只能处理列，不能处理行
Y3 = corr2d(X.t(), K)
print(Y3)

"""
以下部分说明已知X和Y，如何去求解核矩阵K
"""
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))  # 这里的X取得刚才45行的矩阵
Y = Y2.reshape((1, 1, 6, 7))  # 这里的Y取的按照[1.0, -1.0]作为卷积核的输出
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))