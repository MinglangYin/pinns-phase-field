"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving Allen-Cahn equation (1D) at 
        equilibrium state.
    
    phi_t = M_phi*a^2*[grad(grad(phi)) + 4*(b/del)**2*phi*(phi-1)*(2*phi-1)] = 0
    with phi(0) = 0 or 0.5
    where:
        b/del= 10 = D

    Analytical solution: phi = 0.5(1-tanh(b/del*x))

    Input: [x]
    Output: [u]
"""

import sys
sys.path.insert(0,'../Utils')

import torch
import torch.nn as nn
import numpy as np
import time

torch.manual_seed(123456)
np.random.seed(123456)

eta = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(1, 10))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        self.net.add_module('linear_layer_2', nn.Linear(10, 10))
        self.net.add_module('tanh_layer_2', nn.Tanh())
        self.net.add_module('linear_layer_3', nn.Linear(10, 1))

    def forward(self, x):
        return self.net(x)

    def loss_total(self, x):
        u = self.net(x)
        u_x = gradients(u, x)[0]
        loss = 0.5*(u_x)**2 - 1/(4*eta**2)*(1-u**2)**2 ### shenjie paper equ 2.5
        return (loss**2).mean()

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def AC_1D(x):
    return np.tanh(x/np.sqrt(2)/eta)

def main():
    ## parameters
    device = torch.device(f"cpu")
    epochs = 1000
    num_train = 15
    num_test = 15
    lr = 0.001

    # Load Data
    x_test = np.linspace(-1, 1, num=num_test).reshape((num_test, 1))
    x_train = np.linspace(-1, 1, num=num_train).reshape((num_train, 1))
    x_train = torch.tensor(x_train[:,], requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test[:,], requires_grad=True, dtype=torch.float32).to(device)
    x_ic = torch.tensor([0], requires_grad=True, dtype=torch.float32).to(device)
    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss = model.loss_total(x_train) + (model(x_ic))**2
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

    ## test
    u_pred = to_numpy(model(x_train))
    x_train = to_numpy(x_train)
    u_sol = to_numpy(AC_1D(x_train))
    print(f'u_pred: {u_pred}')
    print(f'u_sol: {u_sol}')
    print(f'mean test error is :{((u_pred-u_sol)**2).mean()}')

if __name__ == '__main__':
    main()


















