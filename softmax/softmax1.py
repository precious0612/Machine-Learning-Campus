import torch
from IPython import display
from d2l import torch as d2l
# 引入的Fashion-MNIST数据集， 并设置数据迭代器的批量大小为256
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""初始化模型参数
和之前线性回归的例子一样，这里的每个样本都将用固定长度的向量表示。 
原始数据集中的每个样本都是 28×28 的图像。 在本节中，我们[将展平每个图像，把它们看作长度为784的向量。] 
在后面的章节中，我们将讨论能够利用图像空间结构的特征， 但现在我们暂时只把每个像素位置看作一个特征。
回想一下，在softmax回归中，我们的输出与类别一样多。 
(因为我们的数据集有10个类别，所以网络输出维度为10)。
 因此，权重将构成一个 784×10 的矩阵， 偏置将构成一个 1×10 的行向量。 
 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0。"""
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
"""
在实现softmax回归模型之前，我们简要回顾一下sum运算符如何沿着张量中的特定维度工作。 
如 :numref:subseq_lin-alg-reduction和 :numref:subseq_lin-alg-non-reduction所述，
 [给定一个矩阵X，我们可以对所有元素求和]（默认情况下）。 也可以只求同一个轴上的元素，
 即同一列（轴0）或同一行（轴1）。 如果X是一个形状为(2, 3)的张量，我们对列进行求和，
  则结果将是一个具有形状(3,)的向量。 当调用sum运算符时，我们可以指定保持在原始张量的轴数，
  而不折叠求和的维度。 这将产生一个具有形状(1, 3)的二维张量。

"""
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# X = torch.normal(0, 1, (2, 5))
#
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 定义损失函数
# 拿出真实标号的预测值
# y = torch.tensor([0, 2]) # 代表两个样本的真实的标号
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 对每个样本三个类别的预测值
# print(y_hat[[0, 1], y]) # [0 1]代表样本标号，取出对应标号的预测值；因为第0个样本对应的真实标号是0，所以取0.1，第1个则取0.5


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])  # 拿出真实标号的预测值

# loss = cross_entropy(y_hat, y)


# 分类精度
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# accuracy(y_hat, y) / len(y)

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 累加器
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# evaluate_accuracy(net, test_iter)

# 训练

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式，isinstance() 函数来判断一个对象是否是一个已知的类型
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)

        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        d2l.plt.draw()
        d2l.plt.pause(0.001)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

# num_epochs = 10
# train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 预测
def predict_ch3(net, test_iter, n=10):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    # print(trues)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    # print(preds)
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])



# 开始训练
if __name__ == "__main__":
    batch_size = 256 #
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    lr = 0.1
    num_epochs = 20
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()

    predict_ch3(net, test_iter)
    d2l.plt.show(block=True)


