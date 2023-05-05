import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import torch

# 创建一个人工数据集，并存储在csv（逗号分隔值）文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 为了处理缺失的数据，典型的方法包括插值和删除，这里考虑插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 第一二列放到input里，第三列放到output里
inputs = inputs.fillna(inputs.mean())  # fillna对所有NaN填一个值（均值）
print(inputs)

# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y, sep="\n")