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

# from plotting import newfig, savefig

torch.manual_seed(1234)
np.random.seed(1234)

eps = 0.0001

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(2, 20))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        for num in range(2,10):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(20, 20))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
        self.net.add_module('linear_layer_60', nn.Linear(20, 1))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        u = self.net(x)
        u_g = gradients(u, x)[0]
        u_t, u_x = u_g[:, :1], u_g[:, 1:]
        u_gg = gradients(u_x, x)[0]
        u_xx = u_gg[:,1:]
        loss = u_t - u_xx*eps - (u - u**3)
        return (loss**2).mean()

    def loss_bc(self, x_b_l_train, x_b_r_train):
        u_b_l_pred = self.net(x_b_l_train)
        u_b_r_pred = self.net(x_b_r_train)
        # u_b_l_pred_x = gradients(u_b_l_pred, x_b_l_train)[0][:,1]
        # u_b_r_pred_x = gradients(u_b_r_pred, x_b_r_train)[0][:,1]
        return ((u_b_l_pred - u_b_r_pred)**2).mean() #+ ((u_b_l_pred_x - u_b_r_pred_x)**2).mean()
        # return (u_b_l_pred**2).mean() + (u_b_r_pred**2).mean()

    def loss_ic(self, x_i_train, u_i_sol):
        u_i_pred = self.net(x_i_train)
        return ((u_i_pred - u_i_sol)**2).mean()

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
    return np.sin(np.pi*x)

if __name__=="__main__":
    epochs = 3000
    lr = 0.001
    cuda=1
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    x_star = np.linspace(-1.0, 1.0, 100).reshape((-1,1))
    t_star = np.linspace(0, 3, 50).reshape((-1,1))

    N = x_star.shape[0]    
    T = t_star.shape[0]

    xx = np.tile(x_star,(1,T))
    tt = np.tile(t_star,(1,N)).T

    x = xx.flatten()[:,None]
    t = tt.flatten()[:,None]

    t_dtrain = x_star.flatten()[:,None]*0
    x_dtrain = x_star.flatten()[:,None]

    t_int = np.tile(t_star[0],(1,N)).T
    x_int = np.tile(x_star,(1,1))

    x_int = x_int.flatten()[:,None]
    t_int = t_int.flatten()[:,None]
    Xint = np.concatenate([t_int,x_int], 1)
    u_i_sol = init_cond(x_int)

    t_bc = np.tile(t_star,(1,1)).T
    x_lbc = np.tile(x_star[0],(1,T))
    x_ubc = np.tile(x_star[-1],(1,T))

    t_bc = t_bc.flatten()[:,None]
    x_lbc = x_lbc.flatten()[:,None]
    x_ubc = x_ubc.flatten()[:,None]

    Xlbc = np.concatenate([t_bc, x_lbc], 1)
    Xubc = np.concatenate([t_bc, x_ubc], 1)
    
    X_train = np.hstack((t, x))

    X_train = torch.tensor(X_train, requires_grad=True, dtype=torch.float32).to(device)
    Xlbc = torch.tensor(Xlbc,  dtype=torch.float32).to(device)
    Xubc = torch.tensor(Xubc,  dtype=torch.float32).to(device)
    Xint = torch.tensor(Xint,  dtype=torch.float32).to(device)
    u_i_sol = torch.tensor(u_i_sol, dtype=torch.float32).to(device)

    # print(X_train.shape)
    # print(Xint.shape)
    # print(Xlbc.shape)
    # print(Xubc.shape)

    # print('Xint', Xint)
    # print('u_i_sol', u_i_sol)
    # np.savetxt('u_i_sol.txt', np.concatenate([x_int, u_i_sol], 1))

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(X_train)
            loss_bc = model.loss_bc(Xlbc, Xubc)
            loss_ic = model.loss_ic(Xint, u_i_sol)
            loss = loss_pde + loss_bc + loss_ic
            # print(f'epoch {epoch}: loss {loss:.6f} loss_pde {loss_pde:.6f}, loss_bc {loss_bc:.6f}, loss_ic {loss_ic:.6f}')
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

    u_pred = model(X_train)

    scipy.io.savemat('pred_res.mat',{'t':t, 'x':x, 'u':to_numpy(u_pred)})

