import torch

# 张量表示一个数值组成的数组，这个数组可能有多个维度
x = torch.arange(12)
print(x)
print(type(x))

# 可以通过张量的shape属性来访问张量的形状和张量中元素的总数
print(x.shape)
print(x.numel())

# 要改变一个张量的形状而不改变元素数量和元素值，我们可以调用reshape函数
X = x.reshape(3, 4)
print(X)

# 使用全0、全1、其他常量或者从特定分布中随机采样的数字
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))

# 通过提供包含数值的Python列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(y)
print(y.shape)

# 常见的标准算术运算符（+、-、*、/和**）都可以被升级为按元素运算
a = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])
print(a + b, a - b, a * b, a / b, a ** b, sep="\n")  # **运算符是求幂运算

# 把多个张量连结在一起
c = torch.arange(12, dtype=torch.float32).reshape((3, 4))
d = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((c, d), dim=0))
print(torch.cat((c, d), dim=1))

# 通过逻辑运算符构建二元张量
print(c == d)

# 对张量中的所有元素进行求和会产生一个只有一个元素的张量
print(c.sum())
# 即使形状不同，仍然可以通过调用广播机制（broadcasting mechanism）来执行按元素操作
e = torch.arange(3).reshape((3, 1))
f = torch.arange(2).reshape((1, 2))
print(e, f, sep="\n")
# e中的第一列三个元素被复制到第二列，f中第一行两个元素被复制到第二行和第三行
print(e + f)

# 可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素
print(c[-1], c[1:3])

# 除读取外，我们还可以通过指定索引来将元素写入矩阵
c[1, 2] = 9
print(c)

# 为多个元素赋值相同的值，只需要索引所有元素，然后为它们赋值
c[0:2, :] = 12
print(c)

# 运行一些操作可能会导致为新结果分配内存
before = id(d)
d = d + c
print(id(d) == before)

# 执行原地操作(对元素进行改写，地址不变
z = torch.zeros_like(d)
print('id(z):', id(z))
z[:] = c + d
print('id(z):', id(z))

# 如果在后续计算中没有重复使用c，我们也可以使用c[:]=c+d或c+=d来减少操作的内存开销
before = id(c)
c += d
print(id(c) == before)

# 转换为Numpy张量
A = c.numpy()
B = torch.tensor(A)
print(type(A), type(B))

# 将大小为1的张量转换为Python标量
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
