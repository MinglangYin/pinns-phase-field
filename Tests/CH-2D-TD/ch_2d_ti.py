"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn equation (1D).
    
    u_t - (w_xx + w_yy) = 0, w = -u_xx + u**3 - u,  x in [-1, 1], t in [0, 1]
    with
        u_x = w_x = u_y = w_y = 0
        u(0, ) = tanh( (0.2-np.sqrt((x[:,1]-0.0)**2 + (x[:,2] - 0.0)**2)) / (np.sqrt(2)*0.03) )

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
        u_t, u_x, u_y = u_g[:,0], u_g[:,1], u_g[:,2]
        u_xx = gradients(u_x, x)[0][:,1]
        u_yy = gradients(u_y, x)[0][:,2]

        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:,1], w_g[:,2]
        w_xx = gradients(w_x, x)[0][:,1]
        w_yy = gradients(w_y, x)[0][:,2]

        loss_1 = u_t - (w_xx + w_yy)
        loss_2 = w + u_xx + u_yy - u**3 + u
        return (loss_1**2).mean() + (loss_2**2).mean()

    # x is on the domain boundary
    def loss_bc(self, x):
        u, w = self.net(x)
        u_g = gradients(u, x)[0]
        u_x, u_y = u_g[:,1], u_g[:,2]

        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:,1], w_g[:,2]
        return (u_x**2).mean() + (u_y**2).mean() + (w_x**2).mean() + (w_y**2).mean()

    def loss_ic(self, x_i, u_i, w_i):
        y = self.net(x_i)
        u_i_pred, w_i_pred = y[:,0], y[:,1]
        return ((u_i_pred - u_i)**2).mean() + ((w_i_pred - w_i)**2).mean()

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

def CH_2D_init(x_init):
    x = Symbol('x')
    y = Symbol('y')
    u = tanh( (0.2-sqrt((x-0.0)**2 + (y - 0.0)**2)) / (sqrt(2)*0.03) )
    u_x = u.diff(x)
    u_y = u.diff(y)
    u_xx = u_x.diff(x)
    u_yy = u_y.diff(y)

    f_u = lambdify(x, y, u, 'numpy')
    f_u_xx = lambdify(x, y, u_xx, 'numpy')
    f_u_yy = lambdify(x, y, u_yy, 'numpy')
    u_init = f_u(x_init[:,1], x_init[:,2])
    w_init = -f_u_xx(x_init[:,1]) - f_u_yy(x_init[:,2]) + u_init**3 - u_init
    return u_init, w_init

def AC_2D_init(x):
    ## bubble
    u_init = np.zeros((x.shape[0]))
    v_init = np.zeros((x.shape[0]))
    phi_init = np.tanh( (0.2-np.sqrt((x[:,1]-0.0)**2 + (x[:,2] - 0.0)**2)) / (np.sqrt(2)*0.03) )
    return u_init, v_init, phi_init

def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    epochs = 10000
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 100
    num_b_train = 50    # boundary sampling points
    num_f_train = 50000   # inner sampling points
    num_i_train = 5000   # initial sampling points 
    num_t_train = 40

    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    x_2d = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    
    ## initialization
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init, w_init = CH_2D_init(xt_init)

    ## save init fig
    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(w_init.reshape(num_x, num_y), interpolation='nearest', cmap='rainbow', 
            # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(xt_init[:,1], xt_init[:,2], 'kx', label = 'Data (%d points)' % (xt_init.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(xt_init.min(), xt_init.max(), 2)[:,None]
    fig.savefig('./w_init.png')

"""
    x_2d_ext = np.tile(x_2d, [num_t,1])
    t_ext = np.zeros((num_t*num_x*num_y, 1))
    for i in range (0, num_t):
        t_ext[i*num_x*num_y:(i+1)*num_x*num_y,:] = t[i]
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)
    
    ## sampling
    id_f = np.random.choice(num_x*num_y*num_t, num_f_train, replace=False)
    id_b = np.random.choice(num_x, num_b_train, replace=False) ## Dirichlet
    id_i = np.random.choice(num_x*num_y, num_i_train, replace=False)
    id_t = np.random.choice(num_t, num_t_train, replace=False)

    ## boundary  
    t_b = t[id_t,:]
    t_b_ext = np.zeros((num_t_train*num_b_train, 1))
    for i in range(0, num_t_train):
        t_b_ext[i*num_b_train:(i+1)*num_b_train,:] = t_b[i]
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T

    x_up_sample = x_up[id_b, :]
    x_dw_sample = x_dw[id_b, :]
    x_l_sample = x_l[id_b, :]
    x_r_sample = x_r[id_b, :]

    x_up_ext = np.tile(x_up_sample, (num_t_train, 1))
    x_dw_ext = np.tile(x_dw_sample, (num_t_train, 1))
    x_l_ext = np.tile(x_l_sample, (num_t_train, 1))
    x_r_ext = np.tile(x_r_sample, (num_t_train, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_i = xt_init[id_i, :]
    xt_f = xt_2d_ext[id_f, :]

    ## set data as tensor and send to device
    xt_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    xt_i_train = torch.tensor(xt_init, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_2d, requires_grad=True, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    w_i_train = torch.tensor(w_init, dtype=torch.float32).to(device)
    xt_l_train = torch.tensor(xt_l, requires_grad=True, dtype=torch.float32).to(device)
    xt_r_train = torch.tensor(xt_r, requires_grad=True, dtype=torch.float32).to(device)
    xt_up_train = torch.tensor(xt_up, requires_grad=True, dtype=torch.float32).to(device)
    xt_dw_train = torch.tensor(xt_dw, requires_grad=True, dtype=torch.float32).to(device)

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
"""
if __name__ == '__main__':
    main()

















