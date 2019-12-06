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
from plotting import newfig, savefig
from sympy import *
from random_field import fftind, gaussian_random_field
from image_gradient import *

# torch.manual_seed(1234)
# np.random.seed(1234)
eps = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('net_linear_1', nn.Linear(3, 30))
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
    def loss_bc(self, x_bh, x_bv):

        y_bh = self.net(x_bh)
        y_bv = self.net(x_bv)

        u_bh, w_bh = y_bh[:,0], y_bh[:,1]
        u_bv, w_bv = y_bv[:,0], y_bv[:,1]

        u_bh_g = gradients(u_bh, x_bh)[0]
        w_bh_g = gradients(w_bh, x_bh)[0]
        u_bv_g = gradients(u_bv, x_bv)[0]
        w_bv_g = gradients(w_bv, x_bv)[0]

        u_bh_y, w_bh_y = u_bh_g[:,2], w_bh_g[:,2]
        u_bv_x, w_bv_x = u_bv_g[:,1], w_bv_g[:,1]
        return (u_bh_y**2).mean() + (w_bh_y**2).mean() + (u_bv_x**2).mean() + (w_bv_x**2).mean()

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
############ bubble#############
def CH_2D_init(x_init):
    x, y = symbols('x y')
    # u = tanh( (0.2-sqrt((x-0.0)**2 + (y - 0.0)**2)) / (sqrt(2)*0.03) )
    u = sin(0.2*x+0.2*y)
    u_x, u_y = u.diff(x), u.diff(y)
    u_xx, u_yy = u_x.diff(x), u_y.diff(y)

    f_u = lambdify([x, y], u, 'numpy')
    f_u_xx = lambdify([x, y], u_xx, 'numpy')
    f_u_yy = lambdify([x, y], u_yy, 'numpy')

    u_init = f_u(x_init[:,1], x_init[:,2])
    w_init = -f_u_xx(x_init[:,1], x_init[:,2]) - f_u_yy(x_init[:,1],x_init[:,2]) + u_init**3 - u_init
    return u_init, w_init

########### Random Field ###########
# def get_w(u, sobol_filter):
#     u_x = sobol_filter.grad_h(u)
#     u_xx = sobol_filter.grad_h(u_x)
#     u_y = sobol_filter.grad_v(u)
#     u_yy = sobol_filter.grad_v(u_y)
#     w = -u_xx - u_yy + u**3 - u
#     return w


def main():
    ## parameters
    cuda=0
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    epochs = 1
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 20
    # num_b_train = 50    # boundary sampling points
    # num_f_train = 50000   # inner sampling points
    # num_i_train = 5000   # initial sampling points 
    # num_t_train = 40

    x = np.linspace(-1, 1, num=num_x)
    y = np.linspace(-1, 1, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    # x_test = np.concatenate((t_grid.flatten()[:,None], x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    x_2d = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)
    
    ## initialization
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init, w_init = CH_2D_init(xt_init)
    # u_init = gaussian_random_field(alpha = 8.0, size = num_x)[None,:][None,:]
    # print(u_init.shape)
    # w_init = 

    ############# save init fig #################
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

    fig, ax = newfig(2.0, 1.1)
    ax.axis('off') 
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    h = ax.imshow(u_init.reshape(num_x, num_y), interpolation='nearest', cmap='rainbow', 
            # extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    fig.colorbar(h)
    ax.plot(xt_init[:,1], xt_init[:,2], 'kx', label = 'Data (%d points)' % (xt_init.shape[0]), markersize = 4, clip_on = False)
    line = np.linspace(xt_init.min(), xt_init.max(), 2)[:,None]
    fig.savefig('./u_init.png')

    #######################################
    x_2d_ext = np.tile(x_2d, [num_t,1])
    t_ext = np.zeros((num_t*num_x*num_y, 1))
    for i in range (0, num_t):
        t_ext[i*num_x*num_y:(i+1)*num_x*num_y,:] = t[i]
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)

    ## boundary  
    t_b = t
    t_b_ext = np.zeros((num_t*num_x, 1))
    for i in range(0, num_t):
        t_b_ext[i*num_x:(i+1)*num_x,:] = t_b[i]
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T

    x_up_ext = np.tile(x_up, (num_t, 1))
    x_dw_ext = np.tile(x_dw, (num_t, 1))
    x_l_ext = np.tile(x_l, (num_t, 1))
    x_r_ext = np.tile(x_r, (num_t, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_bh = np.vstack((xt_up, xt_dw))
    xt_bv = np.vstack((xt_l, xt_r))

    xt_i = xt_init
    xt_f = xt_2d_ext

    ## set data as tensor and send to device
    x_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)

    x_i_train = torch.tensor(xt_i, requires_grad=True, dtype=torch.float32).to(device)
    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    w_i_train = torch.tensor(w_init, dtype=torch.float32).to(device)
    
    x_bh_train = torch.tensor(xt_bh, requires_grad=True, dtype=torch.float32).to(device)
    x_bv_train = torch.tensor(xt_bv, requires_grad=True, dtype=torch.float32).to(device)

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
            loss_bc = model.loss_bc(x_bh_train, x_bv_train)
            loss_ic = model.loss_ic(x_i_train, u_i_train, w_i_train)
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
    phi_test = np.zeros((num_t, num_x, num_y))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x*num_y, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
        y_pred = model(xt_tensor)
        phi_test[i,:,:] = to_numpy(y_pred[:, 0]).reshape(num_x, num_y)

        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(phi_test[i,:,:], interpolation='nearest', cmap='rainbow', 
                # extent=[t.min(), t.max(), x.min(), x.max()], 
                    origin='lower', aspect='auto')
        fig.colorbar(h)
        ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt.min(), xt.max(), 2)[:,None]
        fig.savefig('./phi_'+str(i+1000)+'.png')

if __name__ == '__main__':
    main()

















