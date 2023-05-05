import os.path
import d2l.torch
import torch
import torchvision.datasets
from torch import nn
from torch.utils import data
d2l.torch.DATA_HUB['hotdog'] = (d2l.torch.DATA_URL+'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.torch.download_extract('hotdog')
print(data_dir)
#创建两个实例来分别读取训练和测试数据集中的所有图像文件。
train_images = torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'))
test_images = torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'))

# 显示前8个正类样本图片和最后8张负类样本图片（图像的大小和纵横比各有不同）
hotdogs = [train_images[i][0] for i in range(8)]
non_hotdogs = [train_images[-i-1][0] for i in range(8)]
d2l.torch.show_images(hotdogs+non_hotdogs,2,8,scale=1.4)

#RGB三个通道的均值为：0.485, 0.456, 0.406，都是从ImageNet数据集中求出来的；RGB三个通道的标准差为：0.229, 0.224, 0.225，也都是从ImageNet数据集中求出来的
## 使用ImageNet数据集中所有图片RGB通道求出的均值和标准差，以标准化当前数据集图片的每个通道
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])#将图片RGB三个通道标准化
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

# 使用在ImageNet上面预训练好的模型
pretrain_net = torchvision.models.resnet18(pretrained=True)
print(pretrain_net.fc)
'''
输出结果：
Linear(in_features=512, out_features=1000, bias=True)
'''
finetuning_net = torchvision.models.resnet18(pretrained=True)
finetuning_net.fc = nn.Linear(in_features=finetuning_net.fc.in_features,out_features=2)
# 只对最后一层的模型做随机初始化
nn.init.xavier_uniform_(finetuning_net.fc.weight)

'''
这里nn.CrossEntropyLoss(reduction="None")和之前的写法不同
默认reduction="mean"，也就是求完一个batch中的所有样例的损失值后做平均
reduction有三个选项，“mean”：做平均；“sum”:求和；"none":什么也不做就得到一个tensor
在代码中，最后使用l.sum().backward()，实际和一开始指定reduction="sum"效果相同
使用sum的话，求得的梯度与batch_size正相关，这里取batch_size=128，lr=5e-5，近似于reduction='mean'后lr=5e-5*128,因为使用mean后会除以样本数从而导致更新权重参数时梯度变小了，需要把学习率增大
'''
def train_fine_tuning(net,learning_rate,batch_size=128,epochs=5,param_group=True):
    train_iter = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(os.path.join(data_dir,'train'),transform=train_transforms),
                                             batch_size=batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(os.path.join(data_dir,'test'),transform=test_transforms),
                                            batch_size=batch_size,shuffle=False)
    #reduction='none'表示直接求出loss，然后再求和，得出样本总loss大小，不是求样本平均loss，因此loss值会与样本数有直接线性关系，从而当batch_size批量样本数目变多时，一般都会增加学习率。样本平均loss一般与样本数目关系不大，当batch_size批量样本数目变多时，一般学习率不会增加

    loss = nn.CrossEntropyLoss(reduction='none')
    devices = d2l.torch.try_all_gpus()
    if param_group:
        #param_group=True表示使用微调
        #表示除最后一层外前面所有层的参数使用微调，最后一层参数学习率是前面所有层参数学习率的10倍
        param_conv2d = [param for name,param in net.named_parameters() if name not in ['fc.weight','fc.bias']]
        optim = torch.optim.SGD([{'params':param_conv2d},
                                 {'params':net.fc.parameters(),
                                 'lr':learning_rate*10}],
                                lr=learning_rate,weight_decay=0.001)
    else:
        # param_group=False表示不使用微调，直接使用当前数据集对目标模型从零开始进行训练
        optim = torch.optim.SGD(net.parameters(), lr=learning_rate,weight_decay=0.001)

    d2l.torch.train_ch13(net, train_iter, test_iter, loss, optim, epochs, devices)


#使用预训练的模型对目标模型进行训练，使用较小的学习率5e-5
train_fine_tuning(finetuning_net,learning_rate=5e-5)

# 不使用微调的对比
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)