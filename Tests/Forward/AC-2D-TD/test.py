import torch
import numpy as np

tmp = np.array([1, 2, 3, 4, 5])
add_on = np.array([3, 3, 3, 3, 3])
device = 'cpu'
x = torch.tensor(tmp, requires_grad=True, dtype=torch.float32).to(device)
add_on = torch.tensor(add_on, requires_grad=True, dtype=torch.float32).to(device)
y = x[:2]

print(x)

z = (x+add_on).mean()
z.backward()
print(x)