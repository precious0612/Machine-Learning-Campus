"""
权重衰退程序
"""
import torch
from torch import nn
from d2l import torch as d2l

# 生成一些数据，训练数据20，测试数据100，容易产生过拟合
n_train, n_test, num_inputs, batch_size = 100, 500, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train) # 从人工数据集产生训练数据
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test) # 从人工数据集产生测测试数据
test_iter = d2l.load_array(test_data, batch_size, is_train=False)


# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定义训练代码实现
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 150, 0.01
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:

            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


# 忽略正则化直接训练

train(lambd=0)

# 使用权重衰减

train(lambd=3)


train(lambd=30)
d2l.plt.show()

# 简洁实现

# def train_concise(wd):
#     net = nn.Sequential(nn.Linear(num_inputs, 1))
#     for param in net.parameters():
#         param.data.normal_()
#     loss = nn.MSELoss()
#     num_epochs, lr = 100, 0.003
#     trainer = torch.optim.SGD([{
#         "params": net[0].weight,
#         'weight_decay': wd}, {
#             "params": net[0].bias}], lr=lr)
#     animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
#                             xlim=[5, num_epochs], legend=['train', 'test'])
#     for epoch in range(num_epochs):
#         for X, y in train_iter:
#             with torch.enable_grad():
#                 trainer.zero_grad()
#                 l = loss(net(X), y)
#             l.backward()
#             trainer.step()
#         if (epoch + 1) % 5 == 0:
#             animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
#                                      d2l.evaluate_loss(net, test_iter, loss)))
#     print('w的L2范数：', net[0].weight.norm().item())
#
# # 这些图看起来和我们从零开始实现权重衰减时的图相同
#
# train_concise(0)

