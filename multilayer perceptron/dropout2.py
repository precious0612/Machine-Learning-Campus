"""
使用框架简洁实现dropout，对比分析不同的dropout层的效果
"""
import torch
from torch import nn
from d2l import torch as d2l

# 构建模型
net1 = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        # nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        # nn.Dropout(),
        nn.Linear(256, 10))

net2 = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        # nn.Dropout(),
        nn.Linear(256, 10))

net3 = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net1.apply(init_weights);
net2.apply(init_weights);
net3.apply(init_weights);


num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net1.parameters(), lr=lr)
d2l.train_ch3(net1, train_iter, test_iter, loss, num_epochs, trainer)

trainer = torch.optim.SGD(net2.parameters(), lr=lr)
d2l.train_ch3(net2, train_iter, test_iter, loss, num_epochs, trainer)

trainer = torch.optim.SGD(net3.parameters(), lr=lr)
d2l.train_ch3(net3, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()

