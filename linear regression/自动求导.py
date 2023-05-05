import torch
# 假设我们想对函数y = 2x⊤x关于列向量x求导。⾸先，我们创建变量x并为其分配⼀个初始值。
x = torch.arange(4.0)
print(x)
# 在我们计算y关于x的梯度之前，我们需要⼀个地⽅来存储梯度
x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad
# 定义函数
y = 2 * torch.dot(x, x)
print(y)
# 通过调⽤反向传播函数来⾃动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
# 对比梯度是否和求导结果相等
print(x.grad == 4 * x)
# 计算一个新的函数
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
z = x.grad
print(z)

# ⾮标量变量的反向传播
"""当y不是标量时，向量y关于向量x的导数的最⾃然解释是⼀个矩阵。对于⾼阶和⾼维的y和x，求导的结果可
以是⼀个⾼阶张量。
然而，虽然这些更奇特的对象确实出现在⾼级机器学习中（包括深度学习中），但当我们调⽤向量的反向计算
时，我们通常会试图计算⼀批训练样本中每个组成部分的损失函数的导数。这⾥，我们的⽬的不是计算微分
矩阵，而是单独计算批量中每个样本的偏导数之和。"""
# 对⾮标量调⽤backward需要传⼊⼀个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例⼦中，我们只想求偏导数的和，所以传递⼀个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)