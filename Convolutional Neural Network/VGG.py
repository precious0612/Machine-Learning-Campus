"""
VGG实现
"""


import torch
from torch import nn
from d2l import torch as d2l

# 定义VGG块，num_convs：卷积层个数；in_channels：输入通道数, out_channels：输出通道数

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        # vgg输入224*224，224+4-3+1=224
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    # 最大池化，让高宽减半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，
# 每个后续模块将输出通道数量翻倍，直到该数字达到512。
# 由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。
# 可以改为Vgg16或19
# 列表中第一个参数是卷积层数量
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    # 初始化输入通道，可以根据实际情况设置
    in_channels = 1
    # 以conv_arch参数构造vgg网络
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels # 更新下一层的输入通道

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

# 因为vgg计算量更大，构建了一个通道较小的网络，也可以用上面的原始网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)


lr, num_epochs, batch_size = 0.05, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
