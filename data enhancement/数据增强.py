import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('C:/Users/MI/Desktop/cat.jpg')
d2l.plt.imshow(img)

# 定义辅助函数apply。 此函数在输入图像img上多次运行图像增广方法aug并显示所有结果
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    d2l.plt.show()

# 水平方向翻转，随机翻转50%概率
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下方向翻转，随机翻转50%概率
apply(img, torchvision.transforms.RandomVerticalFlip())

# 随机剪裁 第一个值是最后的尺寸  scale：裁剪图片比例  ratio：高宽比 1：2 或 2：1
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)

# 随机改变亮度  brightness亮度, contrast对比度, saturation饱和度, hue色调
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))


apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))


color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 合并增强
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)

