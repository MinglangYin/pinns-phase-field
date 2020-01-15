"""
Author: Minglang Yin
    PINNs (physical-informed neural network) for solving time-dependent Allen-Cahn Navier-Stokes equation (2D).
    
    Governing equtions (Latex, refer to Xiaoning Zheng et al. Interface):
        \frac{\partial\phi}{\partial t} + u\cdot(\nabla\phi) - \tau(-\lambda\Delta\phi + \lambda\gamma f(\phi)) = 0,
        \rho(\frac{\partial u}{\partial t} + (u \cdot \nabla) u) + \nabla p - \mu\Delta u + 
                \lambda\nabla\cdot(\nabla\phi \otimes \nabla\phi) + \frac{(1-\phi)u}{\kappa}= 0,
        \nabla \cdot u = 0
        f(\phi) = F'(\phi) = 1/(h^2)2\phi(\phi-1)(2\phi-1)

    where:
    Input: [t, x, y]
    Output: [u, v, p, phi, psi_1, psi_2]

    Boundary Condition:
        Left: Inflow: velocity: u = , v = 0
        Right: Outflow: n*grad(u)=0
        Up: u=v=0, n*grad(phi)=n*grad(psi)=0
        Down: u=v=0, n*grad(phi)=n*grad(psi)=0

    Note:
        Here we keep the direction of Jacobian the same as convention. 
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

from plotting import newfig, savefig

torch.manual_seed(123456)
np.random.seed(123456)

rho =1 
eta = 1
lmd_e = 0
lmd = 1
K = 0.01
gamma = 1
rho = 1
tau = 1
h = 1

mu = 0.01
Gamma = 0.01
Eta = 0.1


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('linear_layer_1', nn.Linear(3, 15))
        self.net.add_module('tanh_layer_1', nn.Tanh())
        # self.net.add_module('lrelu_layer_1', nn.LeakyReLU())
        for num in range(2,15):
            self.net.add_module('linear_layer_%d' %(num), nn.Linear(15, 15))
            self.net.add_module('tanh_layer_%d' %(num), nn.Tanh())
            # self.net.add_module('lrelu_layer_%d' %(num), nn.LeakyReLU())
        self.net.add_module('linear_layer_50', nn.Linear(15, 4))

    def forward(self, x):
        return self.net(x)

    def loss_pde(self, x):
        y = self.net(x)
        u, v, p, phi = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        u_g, v_g, p_g, phi_g = gradients(u, x)[0], gradients(v, x)[0], gradients(p, x)[0], gradients(phi, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        v_t, v_x, v_y = v_g[:, 0], v_g[:, 1], v_g[:, 2]
        p_x, p_y = p_g[:, 1], p_g[:, 2]
        phi_t, phi_x, phi_y = phi_g[:, 0], phi_g[:, 1], phi_g[:, 2]
   
        u_xx, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_y, x)[0][:, 2]
        v_xx, v_yy = gradients(v_x, x)[0][:, 1], gradients(v_y, x)[0][:, 2]
        phi_xx, phi_xy, phi_yy = gradients(phi_x, x)[0][:, 1], gradients(phi_x, x)[0][:, 2], gradients(phi_y, x)[0][:, 2]
        
        ## Navier Stokes
        # res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - eta*(u_xx + u_yy) \
        #             + lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y) + eta*(1-phi)*u/K
        # res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - eta*(v_xx + v_yy) \
        #             + lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy) + eta*(1-phi)*v/K

        res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) \
                    + lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y) + mu*(1-phi)*u/K
        res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) \
                    + lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy) + mu*(1-phi)*v/K
        res_incomp = u_x + v_y

        ###phi
        F_g = 2/(h**2)*phi*(phi-1)*(2*phi-1)
        # F_g = phi**3 - phi
        w = -Eta**2*(phi_xx + phi_yy) + F_g # + lmd_e/2*(psi_1_x**2+psi_1_y**2 + psi_2_x**2+psi_2_y**2)

        # res_pf = phi_t + (u*phi_x + v*phi_y) + tau*w
        res_pf = phi_t + Eta**2*(u*phi_x + v*phi_y) + Gamma*w

        return (res_incomp**2).mean() + (res_ns_x**2).mean() + (res_ns_y**2).mean() + (res_pf**2).mean()

    def loss_bc(self, x_l, x_r, x_up, x_dw, u_l_real):
        
        ## Periodic
        y_l, y_r, y_up, y_dw = self.net(x_l), self.net(x_r), self.net(x_up), self.net(x_dw)

        u_l, v_l, p_l, phi_l = y_l[:, 0], y_l[:, 1], y_l[:, 2], y_l[:, 3]
        u_r, v_r, p_r, phi_r = y_r[:, 0], y_r[:, 1], y_r[:, 2], y_r[:, 3]
        u_up, v_up, phi_up = y_up[:, 0], y_up[:, 1], y_up[:, 3]
        u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3] 

        phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]
        return ((u_l - u_r)**2).mean() + ((v_l - v_r)**2).mean() + ((phi_l - phi_r)**2).mean() + \
               (u_up**2).mean() + (u_dw**2).mean() + (v_up**2).mean() + (v_dw**2).mean() +\
               (phi_dw_y**2).mean() + (phi_up_y**2).mean() #+ ((p_l - p_r)**2).mean()

        ## velocity inflow
        # y_l, y_r, y_up, y_dw = self.net(x_l), self.net(x_r), self.net(x_up), self.net(x_dw)
        
        # u_l, v_l = y_l[:, 0], y_l[:, 1]
        # u_r, phi_r = y_r[:, 0], y_r[:, 3]
        # u_up, v_up, phi_up = y_up[:, 0], y_up[:, 1], y_up[:, 3]
        # u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3]

        # u_r_x = gradients(u_r, x_r)[0][:, 1]
        # phi_r_x = gradients(phi_r, x_r)[0][:, 1]
        # phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        # phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]

        # return ((u_l - u_l_real)**2).mean() + (v_l**2).mean()\
        #         + (u_r_x**2).mean() + (phi_r_x**2).mean() \
        #         + (u_up**2).mean() + (v_up**2).mean() + (phi_up_y**2).mean()\
        #         + (u_dw**2).mean() + (v_dw**2).mean() + (phi_dw_y**2).mean()

    def loss_ic(self, x_i, u_i, v_i, phi_i):
        y_pred = self.net(x_i)
        u_i_pred, v_i_pred, phi_i_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 3]
        # print(f'u_loss: {((u_i_pred-u_i)**2).mean():6f}')
        # print(f'v_loss: {((v_i_pred-v_i)**2).mean():6f}')
        # print(f'phi_loss: {((phi_i_pred-phi_i)**2).mean():6f}')
        return ((u_i_pred-u_i)**2).mean() + ((v_i_pred-v_i)**2).mean() + ((phi_i_pred-phi_i)**2).mean()

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

def AC_2D_init(x):
    ## bubble (psi field is very tricky)
    u_init = np.zeros((x.shape[0]))
    u_init = 10*(-x[:, 2]**2 + 0.25**2) ##assume the symmetry of the channel
    v_init = np.zeros((x.shape[0]))
    phi_init = (-np.tanh( (0.2-np.sqrt((x[:,1] + 0.0)**2 + (x[:,2] + 0.0)**2)) / (np.sqrt(2)*0.09)) + 1)/2
    return u_init, v_init, phi_init

def left_bound_vel(x):
    return 2*(-x[:, 2]**2 + 0.25**2)

def main():
    ## parameters
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    epochs = 20000
    lr = 0.001

    num_x = 100
    num_y = 100
    num_t = 10
    num_b_train = 100    # boundary sampling points
    num_f_train = 70000   # inner sampling points
    num_i_train = 10000   # initial sampling points 
    num_t_train = 10
    path_list = ['./Predictions/u/', './Predictions/v/', './Predictions/p/', './Predictions/phi/']

    x_min, x_max = -1, 1
    y_min, y_max = -0.25, 0.25
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    x = np.linspace(x_min, x_max, num=num_x)
    y = np.linspace(y_min, y_max, num=num_y)
    t = np.linspace(0, 5, num=num_t)[:, None]
    x_grid, y_grid = np.meshgrid(x, y)
    x_2d = np.concatenate((x_grid.flatten()[:,None], y_grid.flatten()[:,None]), axis=1)

    x_in_grid, y_in_grid = np.meshgrid(x[1:-1], y[1:-1])
    x_in_2d = np.concatenate((x_in_grid.flatten()[:,None], y_in_grid.flatten()[:,None]), axis=1)

    ## initial condition
    xt_init = np.concatenate((np.zeros((num_x*num_y, 1)), x_2d), axis=1)
    u_init, v_init, phi_init = AC_2D_init(xt_init)

    inits = np.zeros((3, num_x*num_y)) # [u, v, phi, psi_1, psi_2]
    inits[0, :] = u_init
    inits[1, :] = v_init
    inits[2, :] = phi_init
    init_list = ['u', 'v', 'phi']
    init_path = './InitConds/'

    ## save init fig
    ## Create Folders
    for path in path_list:
        os.makedirs(path, exist_ok=True)
    os.makedirs(init_path, exist_ok=True)

    ## save init fig
    for i in range(0, 3):
        fig, ax = newfig(2.0, 1.1)
        ax.axis('off') 
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        h = ax.imshow(inits[i, :].reshape(num_x, num_y), interpolation='nearest', cmap='rainbow', origin='lower', aspect=asp_ratio)
        fig.colorbar(h)
        ax.plot(xt_init[:,1], xt_init[:,2], 'kx', label = 'Data (%d points)' % (xt_init.shape[0]), markersize = 4, clip_on = False)
        line = np.linspace(xt_init.min(), xt_init.max(), 2)[:,None]
        fig.savefig(init_path + init_list[i])

    ## inner points
    x_2d_ext = np.tile(x_in_2d, [num_t-1,1])
    t_ext = np.zeros(((num_t-1)*(num_x-2)*(num_y-2), 1))
    for i in range (1, num_t):
        t_ext[(i-1)*(num_x-2)*(num_y-2):i*(num_x-2)*(num_y-2),:] = t[i]
    xt_2d_ext = np.concatenate((t_ext, x_2d_ext), axis=1)
    
    ## sampling
    # id_f = np.random.choice(num_x*num_y*num_t, num_f_train, replace=False)
    # id_b = np.random.choice(num_x, num_b_train, replace=False) ## Dirichlet
    # id_i = np.random.choice(num_x*num_y, num_i_train, replace=False)
    # id_t = np.random.choice(num_t, num_t_train, replace=False)

    ## boundary  
    t_b = t #[id_t,:]
    t_b_ext = np.zeros((num_t_train*num_b_train, 1))
    for i in range(0, num_t_train):
        t_b_ext[i*num_b_train:(i+1)*num_b_train,:] = t_b[i]

    ###########
    x_up = np.vstack((x_grid[-1,:], y_grid[-1,:])).T
    x_dw = np.vstack((x_grid[0,:], y_grid[0,:])).T
    x_l = np.vstack((x_grid[:, 0], y_grid[:, 0])).T
    x_r = np.vstack((x_grid[:, -1], y_grid[:, -1])).T
    # x_bound = np.vstack((x_up, x_dw, x_l, x_r))

    x_up_sample = x_up #[id_b, :]
    x_dw_sample = x_dw #[id_b, :]
    x_l_sample = x_l #[id_b, :]
    x_r_sample = x_r #[id_b, :]

    x_up_ext = np.tile(x_up_sample, (num_t_train, 1))
    x_dw_ext = np.tile(x_dw_sample, (num_t_train, 1))
    x_l_ext = np.tile(x_l_sample, (num_t_train, 1))
    x_r_ext = np.tile(x_r_sample, (num_t_train, 1))

    xt_up = np.hstack((t_b_ext, x_up_ext))
    xt_dw = np.hstack((t_b_ext, x_dw_ext))
    xt_l = np.hstack((t_b_ext, x_l_ext))
    xt_r = np.hstack((t_b_ext, x_r_ext))

    xt_i = xt_init #[id_i, :]
    xt_f = xt_2d_ext #[id_f, :]

    #############left boundary velocity
    u_l = left_bound_vel(xt_l)

    ## set data as tensor and send to device
    xt_f_train = torch.tensor(xt_f, requires_grad=True, dtype=torch.float32).to(device)
    # x_test = torch.tensor(xt_2d_ext, requires_grad=True, dtype=torch.float32).to(device)
    xt_i_train = torch.tensor(xt_init, requires_grad=True, dtype=torch.float32).to(device)
    x_i_train = torch.tensor(x_2d, requires_grad=True, dtype=torch.float32).to(device)

    u_i_train = torch.tensor(u_init, dtype=torch.float32).to(device)
    ul_train = torch.tensor(u_l, dtype=torch.float32).to(device)
    v_i_train = torch.tensor(v_init, dtype=torch.float32).to(device)
    phi_i_train = torch.tensor(phi_init, dtype=torch.float32).to(device)
    
    xt_l_train = torch.tensor(xt_l, requires_grad=True, dtype=torch.float32).to(device)
    xt_r_train = torch.tensor(xt_r, requires_grad=True, dtype=torch.float32).to(device)
    xt_up_train = torch.tensor(xt_up, requires_grad=True, dtype=torch.float32).to(device)
    xt_dw_train = torch.tensor(xt_dw, requires_grad=True, dtype=torch.float32).to(device)

    ## instantiate model
    model = Model().to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    # training
    loss_his = np.zeros((epochs,1))

    def train(epoch):
        model.train()
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train, ul_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train, v_i_train, phi_i_train)
            loss = loss_pde + loss_bc + 30*loss_ic
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}', flush=True)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        loss_his[epoch,0] = loss_value
        # print(f'epoch {epoch}: loss {loss_value:.6f}')

    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {int((toc-tic)/60)} min', flush=True)

    ## test
    Predictions = np.zeros((4, num_t, num_x, num_y))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x*num_y, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float32).to(device)
        y_pred = model(xt_tensor)
        for j in range(0, 4):
            Predictions[j, i, :, :] = to_numpy(y_pred[:, j]).reshape(num_x, num_y)

    print('here 1')

    for i in range(0, num_t):
        print(f'Saving Step: {i}')
        for j in range(0, 4):
            fig, ax = newfig(2.0, 1.1)
            ax.axis('off') 
            gs0 = gridspec.GridSpec(1, 2)
            gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
            h = ax.imshow(Predictions[j,i,:,:], interpolation='nearest', cmap='rainbow', origin='lower', aspect=asp_ratio)
            fig.colorbar(h)
            ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)
            line = np.linspace(xt.min(), xt.max(), 2)[:,None]
            fig.savefig(path_list[j]+str(i+1000)+'.png')
            plt.close()

if __name__ == '__main__':
    main()


















