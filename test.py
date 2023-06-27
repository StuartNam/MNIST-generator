import torch
import math

sum = 0
for i in range(100000):
    a = torch.randn(784)
    b = torch.randn(784)

    sum += (a - b) ** 2

sum /= 100000

print(sum)

