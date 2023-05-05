# 逻辑回归-线性可分

"""案例：根据学生的两门学生成绩，预测该学生是否会被大学录取
数据集：ex2data1.tx"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:/Users/MI/Downloads/ex2data1.txt'
# Exam1和Exam2分别是两门课程的分数，Accepted是是否录取的标记，1代表录取
data = pd.read_csv(path, names=['Exam 1', 'Exam 2', 'Accepted'])

# 画出散点图
fig,ax = plt.subplots()
# 绘制没有被录取的散点图，横轴代表exam1的分数，纵轴代表exam2的分数
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
# 绘制被录取的散点图，横轴代表exam1的分数，纵轴代表exam2的分数
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='exam1',ylabel='exam2')
plt.show()

# 构造数据集，x和y
# 定义获取数据的函数
def get_Xy(data):
    data.insert(0, 'ones', 1)  # 添加一列 样本变为[1 x]
    X_ = data.iloc[:, 0:-1]  # 取前三列作为输入样本
    X = X_.values  # 转换为数组

    y_ = data.iloc[:, -1]  # 取最后一列作为真实值
    y = y_.values.reshape(len(y_), 1)  # 转换为numpy数据，重新变化维度为矩阵

    return X, y

X,y = get_Xy(data) # 调用函数获取数据
print(X.shape, y.shape)

# 定义联系函数，sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义损失函数
def costFunction(X, y, theta):
    A = sigmoid(X @ theta)  # 算出y的值

    first = y * np.log(A)  # 上式中的第一部分
    second = (1 - y) * np.log(1 - A)  # 上式中的第二部分

    return -np.sum(first + second) / len(X)

# 初始化theta的值，并求出损失函数初始值
theta = np.zeros((3,1)) # 给theta赋初始值
theta.shape
cost_init = costFunction(X,y,theta) # 算出初始损失值
print(cost_init) # 打印初始损失值

# 定义梯度下降函数
def gradientDescent(X, y, theta, iters, alpha):
    m = len(X)  # 求出样本个数
    costs = []  # 定义空列表存储每次算出来的损失

    for i in range(iters):
        A = sigmoid(X @ theta)  # 先计算g（X@theta），g为sigmoid函数即联系函数
        theta = theta - (alpha / m) * X.T @ (A - y)
        cost = costFunction(X, y, theta)
        costs.append(cost)
        if i % 1000 == 0:
            print(cost)
    return costs, theta

# 初始化 alpha和iters。后续自行改变参数，求解你认为的最好模型

alpha = 0.004
iters=200000


# 训练数据，通过调用梯度下降函数
costs,theta_final =  gradientDescent(X,y,theta,iters,alpha)
print(theta_final)

# 定义预测函数
def predict(X, theta):
    prob = sigmoid(X @ theta)
    z = []
    for x in prob:
        # print(x)
        if x >= 0.5:
            z.append(1)
        else:
            z.append(0)
    return z


# 计算准确率， 通过自行修改参数找到最大的准确率
y_ = np.array(predict(X,theta_final))
y_pre = y_.reshape(len(y_),1)
acc  = np.mean(y_pre == y)
print(acc)

coef1 = - theta_final[0,0] / theta_final[2,0]
coef2 = - theta_final[1,0] / theta_final[2,0]

# 画出决策边界
x = np.linspace(20,100,100) # x代表的即为x1，f代表的原有的x2
f = coef1 + coef2 * x

fig,ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='exam1',
          ylabel='exam2')

ax.plot(x,f,c='g')
plt.show()



