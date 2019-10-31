"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - w_xx = 0, w = u_xx + u**3 - u,  x in [-1, 1], t in [0, 1]
    with
        u(0, x) = x^2*cos(pi*x)
        u(t, -1) = u(t, 1)
        u_x(t, -1) = u_x(t, 1)

    Input: [t, x]
    Output: [u]
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

from plotting import newfig, savefig

torch.manual_seed(1234)
np.random.seed(1234)

eps = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 20))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,5):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(20, 20))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
        self.net.add_module('linear_layer_50', nn.Linear(20, 1))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:,0], u_g[:,1]
        
        u_gg = gradients(u_x, x)[0]
        u_xx = u_gg[:,1]

        w = u_xx - u**3 + u
        w_x = gradients(w, x)[0][:,1]
        w_xx = gradients(w_x, x)[0][:,1]
        
        loss = u_t + w_xx
        return (loss**2).mean()

    def loss_f(self, x, u_f_train):
        u_f_pred = self.net(x)
        return ((u_f_pred - u_f_train)**2).mean()

    def loss_bc(self, xl_train, xr_train):
        ul = self.net(xl_train)
        ur = self.net(xr_train)
        ul_x = gradients(ul, xl_train)[0][:,1]
        ur_x = gradients(ur, xr_train)[0][:,1]
        ul_xx = gradients(ul_x, xl_train)[0][:,1]
        ur_xx = gradients(ur_x, xr_train)[0][:,1]

        wl = ul_xx - 1/eps*(ul**3 - ul)
        wr = ur_xx - 1/eps*(ur**3 - ur)
        wl_x = gradients(wl, xl_train)[0][:,1]
        wr_x = gradients(wr, xr_train)[0][:,1]
        
        return (ul_x**2).mean() + (ur_x**2).mean() + (wl_x**2).mean() + (wr_x**2).mean()

    def loss_ic(self, xi_train, ui_train):
        ui = self.net(xi_train)
        return ((ui - ui_train)**2).mean()

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

def init_cond(x):
    return np.tanh(2*np.pi*x[:,1])
    # return 1+np.cos(np.pi*x)

def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    epochs = 10000
    num_b = 100
    num_i = 100
    num_f = 1000
    num_x = 100
    num_t = 100
    lr = 0.001
    
    ## x: array, x_grid: grid data, X: flatten data
    X = np.linspace(-1, 1, num_x)[:,None]
    T = np.linspace(0, 5, num_t)[:,None]
    t_grid, x_grid = np.meshgrid(T, X)
    
    TT = t_grid.flatten()[:, None]
    XX = x_grid.flatten()[:, None]
    x = np.concatenate((TT, XX), axis=1)
    xi = np.concatenate( ( np.zeros((num_x, 1)), X), axis=1)
    xl = np.concatenate((T, X[0]*np.ones((num_t, 1))), axis=1)
    xr = np.concatenate((T, X[-1]*np.ones((num_t, 1))), axis=1)
    
    ## Initial, Boundary and Inner data
    id_i = np.random.choice(num_x, num_i, replace=False)
    id_b = np.random.choice(num_t, num_b, replace=False)
    id_f = np.random.choice(x.shape[0], num_f, replace=False)
    xi_train = xi[id_i]
    xl_train = xl[id_b]
    xr_train = xr[id_b]
    xf_train = x[id_f]
    x_test = x
    ui_train = init_cond(xi_train)

    ## Form data tensor and send
    xi_train = torch.tensor(xi_train, dtype=torch.float32).to(device)
    xl_train = torch.tensor(xl_train, requires_grad=True, dtype=torch.float32).to(device)
    xr_train = torch.tensor(xr_train, requires_grad=True, dtype=torch.float32).to(device)
    xf_train = torch.tensor(xf_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    ui_train = torch.tensor(ui_train, dtype=torch.float32).to(device)

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
            loss_ic = model.loss_ic(xi_train, ui_train)
            loss = loss_pde + loss_bc + loss_ic
            print(f'epoch {epoch}: loss {loss:.6f} loss_pde {loss_pde:.6f}, loss_bc {loss_bc:.6f}, loss_ic {loss_ic:.6f}')
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
    u_test = model(x_test)
    u_test = to_numpy(u_test.view(num_t, num_x))
    for t in range(num_t):
        print(f'printing snapshot {t}')
        fig, ax = newfig(2.0, 1.1)
        plt.plot(X, u_test[t,:])
        fig.savefig('./u_'+str(t+1000)+'.png')
        plt.close('all')

if __name__ == '__main__':
    main()


















