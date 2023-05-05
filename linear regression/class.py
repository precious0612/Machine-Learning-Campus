import torch

# 使用GPU（M1）加速
device = torch.device('mps')

vector_alpha = torch.Tensor([1, 2, 3, 4]).to(device)
vector_beta = torch.Tensor([5, 6, 7, 8]).to(device)

print(vector_alpha.reshape([2, 2]))

print(vector_alpha.reshape([2, 2]) + vector_beta.reshape([2, 2]))
print(vector_alpha.reshape([2, 2]) - vector_beta.reshape([2, 2]))
print(vector_alpha.reshape([2, 2]) * vector_beta.reshape([2, 2]))
print(vector_alpha.reshape([2, 2]) / vector_beta.reshape([2, 2]))

print(vector_alpha.reshape([2, 2]).inverse())
