
online_clients=[2,0,1]
nets_list=[[5],[2],[6]]

for net_id, _ in zip(online_clients, nets_list):
    net = nets_list[net_id]

for net_id in online_clients:
    net = nets_list[net_id]

print(nets_list)
from tqdm import tqdm

iterator = tqdm(range(40))
cut  = 0.2
weight = (0.5 - cut) / 0.5 ** 3
for i in iterator:
    target_cos =   1 - (weight * (i / 40  - 0.5) ** 3 + 0.5)
    print(target_cos)

import torch.nn as nn
import torch

loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
print(output)
output.backward()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
for i in range(2):
    print(torch.bernoulli(a))

print(torch.bernoulli(a))

import torch

x = torch.FloatTensor(torch.rand([10]))
print('x', x)
y = torch.FloatTensor(torch.rand([10]))
print('y', y)

similarity = torch.cosine_similarity(x, y, dim=0)

inner = torch.mm(x.reshape(1,-1),y.reshape(-1,1))
x_len = torch.norm(x, p='fro', dim=None, keepdim=False, out=None, dtype=None)
y_len = torch.norm(y, p='fro', dim=None, keepdim=False, out=None, dtype=None)
cos = inner/(x_len * y_len)

print('similarity', similarity)

