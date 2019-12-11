"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - w_xx = 0, w = -u_xx + u**3 - u,  x in [-1, 1], t in [0, 1]
    with
        u(0, x) = tanh(x)
        w(0, x) = -u_xx + u**3 - u

    Input: [t, x]
    Output: [u]

    Testing for network structure
    Neurons number: 2-20-20-20-20-2
"""

import sys
sys.path.insert(0,'../../Utils')

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
eps = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('net_linear_1', nn.Linear(2, 30))
        self.net.add_module('net_relu_1', nn.ReLU())
        for num in range(2,6):
            self.net.add_module('net_linear_%d' %(num), nn.Linear(30, 30))
            self.net.add_module('net_relu_%d' %(num), nn.ReLU())
        self.net.add_module('net_linear_50', nn.Linear(30, 2))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        y = self.net(x)
        u, w = y[:,0], y[:,1]

        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:,0], u_g[:,1]
        u_xx = gradients(u_x, x)[0][:,1]
        w_x = gradients(w, x)[0][:,1]
        w_xx = gradients(w_x, x)[0][:,1]

        loss_1 = u_t - w_xx
        loss_2 = w + u_xx - u**3 + u
        return (loss_1**2).mean() + (loss_2**2).mean()

    def loss_bc(self, xl_train, xr_train):
        yl = self.net(xl_train)
        yr = self.net(xr_train)
        ul, wl = yl[:,0], yl[:,1]
        ur, wr = yr[:,0], yr[:,1]

        wl_x = gradients(wl, xl_train)[0][:,1]
        wr_x = gradients(wr, xr_train)[0][:,1]
        ul_x = gradients(ul, xl_train)[0][:,1]
        ur_x = gradients(ur, xr_train)[0][:,1]        
        return (ul_x**2).mean() + (ur_x**2).mean() + (wl_x**2).mean() + (wr_x**2).mean()

    def loss_ic(self, xi_train, ui_train, wi_train):
        y = self.net(xi_train)
        ui, wi = y[:,0], y[:,1]
        return ((ui - ui_train)**2).mean() + ((wi - wi_train)**2).mean()

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
    u = tanh(x)
    u_x = u.diff(x)
    u_xx = u_x.diff(x)
    
    f_u = lambdify(x, u, 'numpy')
    f_u_xx = lambdify(x, u_xx, 'numpy')
    u_init = f_u(x_init[:,1])
    w_init = -f_u_xx(x_init[:,1]) + u_init**3 - u_init
    return u_init, w_init

# def init_cond(x_init):
#     length = x_init.shape[0]
#     u_init = np.ones((length, 1))
#     w_init = u_init**3 - u_init
#     return u_init, w_init

def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    epochs = 5000
    num_b = 100
    num_i = 100
    num_f = 10000
    num_x = 100
    num_t = 100
    lr = 0.001
    
    ## x: array, x_grid: grid data, X: flatten data
    X = np.linspace(-5, 5, num_x)[:,None]
    T = np.linspace(0, 50, num_t)[:,None]
    t_grid, x_grid = np.meshgrid(T, X)
    
    TT = t_grid.flatten()[:, None]
    XX = x_grid.flatten()[:, None]
    x = np.concatenate((TT, XX), axis=1)
    xi = np.concatenate( ( np.zeros((num_x, 1)), X), axis=1)
    xl = np.concatenate((T, X[0]*np.ones((num_t, 1))), axis=1)
    xr = np.concatenate((T, X[-1]*np.ones((num_t, 1))), axis=1)
    ui, wi = init_cond(xi)

    ## Initial, Boundary and Inner data
    id_i = np.random.choice(num_x, num_i, replace=False)
    id_b = np.random.choice(num_t, num_b, replace=False)
    id_f = np.random.choice(x.shape[0], num_f, replace=False)
    
    xi_train = xi[id_i]
    ui_train = ui[id_i]
    wi_train = wi[id_i]

    xl_train = xl[id_b]
    xr_train = xr[id_b]
    
    xf_train = x[id_f]
    x_test = x

    ## Take all points into training
    # xi_train = xi
    # ui_train = ui
    # wi_train = wi
    # xl_train = xl
    # xr_train = xr
    # xf_train = x
    # x_test = x
    
    ## printing initial data
    fig, ax = newfig(2.0, 1.1)
    plt.plot(X, ui)
    fig.savefig('./u_init.png')
    plt.close('all')

    fig, ax = newfig(2.0, 1.1)
    plt.plot(X, wi)
    fig.savefig('./w_init.png')
    plt.close('all')

    ## Form data tensor and send
    xi_train = torch.tensor(xi_train, dtype=torch.float32).to(device)
    xl_train = torch.tensor(xl_train, requires_grad=True, dtype=torch.float32).to(device)
    xr_train = torch.tensor(xr_train, requires_grad=True, dtype=torch.float32).to(device)
    xf_train = torch.tensor(xf_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    ui_train = torch.tensor(ui_train, dtype=torch.float32).to(device)
    wi_train = torch.tensor(wi_train, dtype=torch.float32).to(device)

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xf_train)
            loss_bc = model.loss_bc(xl_train, xr_train)
            loss_ic = model.loss_ic(xi_train, ui_train, wi_train)
            loss = loss_pde + loss_bc + 10*loss_ic
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

    ## plotting and testing
    u_test = to_numpy(model(x_test)[:,0])
    u_test = np.reshape(u_test, [num_x, num_t]).T
    for t in range(num_t):
        print(f'printing snapshot {t}')
        fig, ax = newfig(2.0, 1.1)
        plt.plot(X, u_test[t,:])
        plt.ylim([-1.5, 1.5])
        fig.savefig('./u_'+str(t+1000)+'.png')
        plt.close('all')

if __name__ == '__main__':
    main()


















