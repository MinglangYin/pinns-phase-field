## This test script is to validate if two sequential model concatenates, w = f(x,t), u = g(w), what dw/dx would be.

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(1234)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net_w = nn.Sequential()
        self.net_w.add_module('1', nn.Linear(2, 1))
        self.net_u = nn.Sequential()
        self.net_u.add_module('2', nn.Linear(1,1))

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(w)
        grad1 = gradients(u, x)[0]
        grad2 = gradients(w, x)[0]
        grad3 = gradients(u, w)[0]
        return grad1, grad2, grad3
    
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


def main():
    device = "cpu"
    x = np.array([0,1])
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32).to(device)
    
    model = Model()
    grad1, grad2, grad3 = model(x)
    print(grad1, grad2, grad3)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

if __name__ == '__main__':
    main()