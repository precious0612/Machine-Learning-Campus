import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取文件，改成你自己保存文件的路径
data = pd.read_csv('data/ex1data2.txt', names=['size', 'bedrooms', 'price'])



# 2.数据预处理：特征归一化
# 定义归一化函数
def normalize_feature(data):
    z = (data - data.mean()) / data.std()
    return z  # （xi-均值）/方差

data = normalize_feature(data) # 调用归一化函数
data.plot.scatter('size','price',label='size')
plt.show()
#
# data.plot.scatter('bedrooms','price',label='bedrooms')
# plt.show()

# 添加全为1的列，主要是添加为[1 X]矩阵
data.insert(0,'ones',1)

# 构造数据集，X为特征向量，也就是输入的X，取数据集的前三列所有行
X = data.iloc[:,0:-1]

# 构造数据集，y取数据集最后一列
y = data.iloc[:,-1]

# 将dataframe转成数组，ndarray
X = X.values
y = y.values
y = y.reshape(47,1) # 将y转换为矩阵

# 损失函数
def costFunction(X,y,theta):
    inner =np.power( X @ theta - y, 2)
    return np.sum(inner) / (2 * len(X))

# 梯度下降函数
def gradientDescent(X, y, theta, alpha, iters, isprint=False):  # isprint是设置的是否打印参数，这里默认为不打印。
    costs = []

    for i in range(iters):
        theta = theta - (X.T @ (X @ theta - y)) * alpha / len(X)
        cost = costFunction(X, y, theta)
        costs.append(cost)

        if i % 100 == 0:
            if isprint:
                print(cost)

    return theta, costs

# 初始化参数，不同alpha下的效果,可以修改
theta = np.zeros((3,1))
candinate_alpha = [0.0001,0.006,0.01,0.015,0.0005,0.1]
iters = 3000

fig, ax = plt.subplots()

for alpha in candinate_alpha:
    [_, costs] = gradientDescent(X, y, theta, alpha, iters)  # 这里要对比不同学习率的损失函数，所以只关注costs
    ax.plot(np.arange(iters), costs, label=alpha)
    ax.legend()

ax.set(xlabel='iters',
       ylabel='cost',
       title='cost vs iters')
plt.show()

