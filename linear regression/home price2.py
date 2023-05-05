# 1. 通过使用深度学习框架来简洁的实现 线性回归模型 生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
import pandas as pd
data1 = pd.read_csv(r"data\ex1data2.txt",names = ['size','rooms','price'])

def normalize_feature(data):
    z = (data - data.mean()) / data.std()
    return z  # （xi-均值）/方差

data1 = normalize_feature(data1) # 调用归一化函数

x = data1.iloc[:,0:-1]
y = data1.iloc[:,-1]
X = x.values
y = y.values
X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32)
features,labels = X,y.reshape(-1,1)

# 构造一个真实的w和b，然后通过人工数据合成函数生成我们需要的features和labels

# 2. 调用框架中现有的API来读取数据
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 5
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# 假设我们已经有features和labels了，我们把他做成一个List传到tensor的dataset里面，把我们的X和y传进去，得到pytorch的一个dataset，（也就是说dataset里面是由两部分组成，features和labels）
# dataset里面拿到数据集之后我们可以调用dataloader函数每次从里面随机挑选batch_size个样本出来，shuffle是指是否需要随机去打乱顺序，如果是train则是需要的
# 构造了一个data_iter（可迭代对象）之后,然后用iter()转成python的一个迭代器,再通过next()函数得到一个X和y

# TensorDataset：把输入的两类数据进行一一对应，一个*表示解包（见python书的P175，一个*表示使用一个已经存在的列表作为可变参数）
# DataLoader：构建可迭代的数据装载器
# enumerate：返回值有两个,一个是序号，一个是数据（包含训练数据和标签）


# 3. 使用框架的预定义好的层
from torch import nn  # nn是neural network的缩写，里面有大量定义好的层
#
net = nn.Sequential(nn.Linear(2, 1))

# 线性回归用的是nn里面的线性层（或者说是全连接层），它唯一要指定的是输入和输出的维度是多少，此处的输入维度是2，输出维度是1
# 线性回归就是简单的单层神经网络，为了以后的方便，放进一个Sequential的容器里面，可以理解为一个list of layers把层按顺序一个一个放在一起

# Sequential是一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
# Sequential 是一个容器，里面可以放置任何层，不一定是线性层

# 4. 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 上面定义的net就是只有一个layer，可以通过索引[0]访问到layer,然后用.weight访问到他的w，用.data访问真实data,normal_表示使用正态分布来替换到data的值，使用的是均值为0方差为0.01来替换
# 偏差直接设置为0

# 5. 计算均方误差使用的是MSELoss类，也称平方L2范数
loss = nn.MSELoss()  # 在nn中均方误差叫MSELoss

# 6. 实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.06)

# SGD至少传入两个参数。net.parameters里面就包括了所有的参数，w和b；指定学习率0.03
# Optimize.Stochastic Gradient Descent （随机梯度下降法）    optim是指optimize优化，sgd是优化的一种方法
# L1范数是算数差，L2范数是算平方差


# 7. 训练过程代码与我们从零开始实现时所做的非常相似
num_epochs = 5 # 迭代3个周期
for epoch in range(num_epochs):
    for X, y in data_iter:  # 在data_iter里面一次一次的把minibatch（小批量）拿出来放进net里面
        l = loss(net(X), y)  # net（）这里本身带了模型参数，不需要把w和b放进去了，net(X)是预测值，y是真实值，拿到预测值和真实值做Loss
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 计算反向传播，这里pytorch已经做了sum就不需要在做sum了（loss是一个张量，求sum之后是标量）
        trainer.step()  # 有了梯度之后调用step（）函数来进行一次模型的更新。调用step函数，从而分别更新权重和偏差
    l = loss(net(features), labels)  # 当扫完一遍数据之后，把所有的feature放进network中，和所有的Label作一次Loss
    print(f'epoch {epoch + 1}, loss {l:f}')  # {l:f} 是指打印l，格式是浮点型



