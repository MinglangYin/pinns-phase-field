"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0, x in [-1, 1], t in [0, 1]
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

torch.manual_seed(123456)
np.random.seed(123456)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 50))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,7):
            self.net.add_module('linear_layer_%d' %(num+1), nn.Linear(50, 50))
            self.net.add_module('tanh_layer_%d' %(num+1), nn.Tanh())
        self.net.add_module('linear_layer_7', nn.Linear(50, 2))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:, 0], u_g[:, 1]
        u_xx = gradients(u_x, x)[0][:, 1]
        loss = u_t - 0.01*u_xx + 5.0*u**3 - 5.0*u
        return (loss**2).mean()

    def loss_f(self, x, u_f_train):
        u_f_pred = self.net(x)
        return ((u_f_pred - u_f_train)**2).mean()

    def loss_bc(self, x_b_l_train, x_b_r_train):
        u_b_l_pred = self.net(x_b_l_train)
        u_b_r_pred = self.net(x_b_r_train)
        u_b_l_pred_x = gradients(u_b_l_pred, x_b_l_train)[0][:,1]
        u_b_r_pred_x = gradients(u_b_r_pred, x_b_r_train)[0][:,1]
        return ((u_b_l_pred - u_b_r_pred)**2).mean() + ((u_b_l_pred_x - u_b_r_pred_x)**2).mean()

    def loss_ic(self, x_i_train, u_i_train):
        u_i_pred = self.net(x_i_train)
        return ((u_i_pred - u_i_train)**2).mean()

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


def main():
    ## parameters
    cuda=1
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    epochs = 10000
    num_b_train = 50
    num_f_train = 500
    lr = 0.001

    ## pre-processing 
    data = scipy.io.loadmat('../../Data/AC/AC.mat')
    
    ## x: array, x_grid: grid data, X: flatten data
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None] 
    t_grid, x_grid = np.meshgrid(t, x)
    exact_grid = np.real(data['uu'])
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]
    Exact = exact_grid.flatten()[:, None]

    ## Initial&Boundary data
    id_i = np.random.choice(x.shape[0], num_b_train, replace=False) 
    id_b = np.random.choice(t.shape[0], num_b_train, replace=False) 
    id_f = np.random.choice(Exact.shape[0], num_f_train, replace=False) 

    x_i = x_grid[id_i, 0][:,None]
    t_i = t_grid[id_i, 0][:,None]
    u_i_train = exact_grid[id_i, 0][:,None]
    x_i_train = np.hstack((t_i, x_i))

    x_b_l = x_grid[0, id_b][:,None]
    x_b_r = x_grid[-1, id_b][:,None]
    t_b_l = t_grid[0, id_b][:,None]
    t_b_r = t_grid[-1, id_b][:,None]
    x_b_l_train = np.hstack((t_b_l, x_b_l))
    x_b_r_train = np.hstack((t_b_r, x_b_r))

    x_f = X[id_f, 0][:,None]
    t_f = T[id_f, 0][:,None]
    x_f_train = np.hstack((t_f, x_f))
    u_f_train = Exact[id_f, 0][:,None]

    x_test = np.hstack((T, X))

    ## Form data tensor and send
    x_i_train = torch.tensor(x_i_train, requires_grad=True, dtype=torch.float32).to(device)
    x_b_l_train = torch.tensor(x_b_l_train, requires_grad=True, dtype=torch.float32).to(device)
    x_b_r_train = torch.tensor(x_b_r_train, requires_grad=True, dtype=torch.float32).to(device)
    x_f_train = torch.tensor(x_f_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_i_train, requires_grad=True, dtype=torch.float32).to(device)
    u_f_train = torch.tensor(u_f_train, requires_grad=True, dtype=torch.float32).to(device)

    # print(f'x_i_train: {x_i_train}')
    # print(f'u_i_train: {u_i_train}')
    # print(f'x_b_l_train: {x_b_l_train}')
    # print(f'x_b_r_train: {x_b_r_train}')
    # print(f'x_f_train: {x_f_train}')
    # print(f'u_f_train: {u_f_train}')
    # print(f'x_test: {x_test}')

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(x_f_train)
            # loss_f = model.loss_f(x_f_train, u_f_train)
            loss_bc = model.loss_bc(x_b_l_train, x_b_r_train)
            loss_ic = model.loss_ic(x_i_train, u_i_train)
            loss = loss_pde + loss_bc + loss_ic
            # loss = loss_f + loss_bc + loss_ic
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

    print(f'u_pred: {to_numpy(model(x_f_train))}')
    print(f'u_train: {to_numpy(u_f_train)}')
    print(f'u_train - u_pred: {((to_numpy(model(x_f_train))-to_numpy(u_f_train))**2).mean()}')

    ## test
    u_test = to_numpy(model(x_test)).reshape(exact_grid.shape[0], exact_grid.shape[1])
    x_test = to_numpy(x_test)
    u_sol = exact_grid
    print(f'mean test error is :{((u_test-u_sol)**2).mean()}')

    ## plotting
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_test.T, interpolation='nearest', cmap='rainbow', 
               # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(x_test[:,1], x_test[:,0], 'kx', label = 'Data (%d points)' % (x_test.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(x_test.min(), x_test.max(), 2)[:,None]
    savefig('./u_test')

    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_sol.T, interpolation='nearest', cmap='rainbow', 
               # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    # ax.plot(x[:,1], x[:,0], 'kx', label = 'Data (%d points)' % (x.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(x_test.min(), x_test.max(), 2)[:,None]
    savefig('./u_sol')

if __name__ == '__main__':
    main()


















