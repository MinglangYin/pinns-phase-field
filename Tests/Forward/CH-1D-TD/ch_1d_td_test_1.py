"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - w_xx = 0, w = -u_xx + u**3 - u,  x in [-1, 1], t in [0, 1]
    with
        u(0, x) = tanh(x)
        w(0, x) = -u_xx + u**3 - u

    Input: [t, x]
    Output: [u, w]

    Testing for network structure
    Neurons number: 2-20-20-20-20-1, (2+1)-20-20-20-20-1

"""

import sys
import os
sys.path.insert(0,'../../../Utils')

import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io

from scipy.misc import derivative
from sympy import *
from plotting import newfig, savefig

# torch.manual_seed(1234)
# np.random.seed(1234)
eps = 0.02

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net_w = nn.Sequential()
        self.net_w.add_module('net_w_linear_1', nn.Linear(2, 20))
        # self.net_w.add_module('net_w_relu_1', nn.ReLU())
        self.net_w.add_module('net_w_tanh_1', nn.Tanh())
        for num in range(2,5):
            self.net_w.add_module('net_w_linear_%d' %(num), nn.Linear(20, 20))
            # self.net_w.add_module('net_w_relu_%d' %(num), nn.ReLU())
            self.net_w.add_module('net_w_tanh_%d' %(num), nn.Tanh())
        self.net_w.add_module('net_w_linear_50', nn.Linear(20, 1))

        self.net_u = nn.Sequential()
        self.net_u.add_module('net_u_linear_1', nn.Linear(2, 20))
        # self.net_u.add_module('net_u_relu_1', nn.ReLU())
        self.net_u.add_module('net_u_tanh_1', nn.Tanh())
        for num in range(2,5):
            self.net_u.add_module('net_u_linear_%d' %(num), nn.Linear(20, 20))
            # self.net_u.add_module('net_u_relu_%d' %(num), nn.ReLU())
            self.net_u.add_module('net_u_tanh_%d' %(num), nn.Tanh())
        self.net_u.add_module('net_u_linear_50', nn.Linear(20, 1))

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        output = torch.cat((u, w), dim=1)
        return output

    def loss_pde(self, x):
        u, w = self.net_u(x), self.net_w(x)
        
        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:,0], u_g[:,1]
        u_xx = gradients(u_x, x)[0][:,1]
        w_x = gradients(w, x)[0][:,1]
        w_xx = gradients(w_x, x)[0][:,1]

        loss_1 = u_t - w_xx
        # loss_2 = w + eps**2*u_xx - u**3 + u
        loss_2 = w + eps**2*u_xx - (u**3 - u)
        print(f'loss 1: {(loss_1**2).mean()}, loss 2: {(loss_2**2).mean()}')
        return (loss_1**2).mean() + (loss_2**2).mean()

    def loss_bc(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        w_x = gradients(w, x)[0][:,1]
        u_x = gradients(u, x)[0][:,1]     
        return (u_x**2).mean() + (w_x**2).mean()

    def loss_ic(self, x, u_init, w_init):
        u = self.net_u(x)
        w = self.net_w(x)
        return ((u - u_init)**2).mean() + ((w - w_init)**2).mean()

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

def init_cond(x_init):
    x = Symbol('x')
    u = tanh(x/(sqrt(2)*0.05) )
    u_x = u.diff(x)
    u_xx = u_x.diff(x)
    
    f_u = lambdify(x, u, 'numpy')
    f_u_xx = lambdify(x, u_xx, 'numpy')
    u_init = f_u(x_init[:,1])
    w_init = -f_u_xx(x_init[:,1])*eps**2 + (u_init**3 - u_init)
    return u_init[:, None], w_init[:, None]

# def init_cond(x):
#     return np.tanh(x[:,1]/(np.sqrt(2)*eps) )

def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    epochs = 4000
    num_x = 100
    num_t = 10
    lr = 0.001

    os.makedirs("Predictions", exist_ok=True)
    
    ##
    x = np.linspace(-1, 1, num=num_x)
    t = np.linspace(0, 1, num=num_t)
    x_grid, t_grid = np.meshgrid(x, t)
    x_f = np.concatenate((t_grid[1:-1,1:-1].flatten()[:,None], x_grid[1:-1,1:-1].flatten()[:,None]), axis=1)
    x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None]), axis=1)

    ##
    x_init = np.concatenate((np.zeros((num_x, 1)), x[:, None]), axis=1 )
    x_b_l = np.concatenate((t_grid[:,0][:,None], x_grid[:,0][:,None]), axis=1)
    x_b_r = np.concatenate((t_grid[:,-1][:,None], x_grid[:,-1][:,None]), axis=1)
    x_b = np.concatenate((x_b_l, x_b_r))

    ## Initial Condition
    u_init, w_init = init_cond(x_init)

    # ## Form data tensor and send
    x_i_train = torch.tensor(x_init, dtype=torch.double).to(device)
    x_b_train = torch.tensor(x_b, requires_grad=True, dtype=torch.double).to(device)
    x_f_train = torch.tensor(x_f, requires_grad=True, dtype=torch.double).to(device)
    x_test = torch.tensor(x_test, dtype=torch.double).to(device)
    u_i_train = torch.tensor(u_init, dtype=torch.double).to(device)
    w_i_train = torch.tensor(w_init, dtype=torch.double).to(device)

    ## instantiate model
    model = Model().to(device)
    model.double()
    
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_f_train)
            loss_bc = model.loss_bc(x_b_train)
            loss_ic = model.loss_ic(x_i_train, u_i_train, w_i_train)
            loss = loss_pde + loss_bc + loss_ic
            print(f'epoch {epoch}: loss_pde {loss_pde:.6f}, loss_bc {loss_bc:.6f}, loss_ic {loss_ic:.6f}')
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...')
    tic = time.time()
    for epoch in range(1, epochs + 1):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {toc-tic}')

    u_pred = to_numpy(model(x_test))
    scipy.io.savemat('ch_data.mat',{'x':to_numpy(x_test), 'u':u_pred})

    u_pred = u_pred[:, 0].reshape(num_t, num_x)
    for tt in range(0, num_t):
        print(f'printing step: {tt}')
        fig, ax = plt.subplots()
        ax.plot(x, u_pred[tt,:])
        fig.savefig('./Predictions/u_'+str(tt+1000)+'.png')
        plt.close()

if __name__ == '__main__':
    main()


















