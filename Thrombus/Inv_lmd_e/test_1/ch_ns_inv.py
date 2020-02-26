"""
Author: Minglang Yin
    PINNs_no_psi (physical-informed neural network) for solving time-dependent Cahn-Hilliard Navier-Stokes equation (2D).

    Governing equtions (Latex):
        \frac{\partial\phi}{dt} + u\cdot(\nabla\phi) + \tau\Delta w = 0, \\
        w = \Delta\phi - f(\phi), \\
        \rho(\frac{\partial u}{\partial t} + (u \cdot \nabla) u) + \nabla p - \mu\Delta u + \lambda(\Delta\phi- f(\phi))\nabla \phi = 0,\\
        \nabla \cdot u = 0, \\
        f(\phi) = F'(\phi) = 1/(h^2)(\phi^3- \phi) 

    where
        Input: [t, x, y]
        Output: [u, v, p, phi], [w, psi_1, psi_2]
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

import matplotlib
matplotlib.use('Agg')

torch.manual_seed(123456)
np.random.seed(123456)

h = 0.05
lmd = 0.5*0.00010607
#lmd_e = 0.5
tau = 1e-6
rho = 1
mu = 0.02
K = 0.01

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net_w = nn.Sequential()
        self.net_w.add_module('net_w_linear_1', nn.Linear(3, 20))
        self.net_w.add_module('net_w_tanh_1', nn.Tanh())
        for num in range(2,10):
            self.net_w.add_module('net_w_linear_%d' %(num), nn.Linear(20, 20))
            self.net_w.add_module('net_w_tanh_%d' %(num), nn.Tanh())
        self.net_w.add_module('net_w_linear_50', nn.Linear(20, 3))

        self.net_u = nn.Sequential()
        self.net_u.add_module('net_u_linear_1', nn.Linear(3, 20))
        self.net_u.add_module('net_u_tanh_1', nn.Tanh())
        for num in range(2,10):
            self.net_u.add_module('net_u_linear_%d' %(num), nn.Linear(20, 20))
            self.net_u.add_module('net_u_tanh_%d' %(num), nn.Tanh())
        self.net_u.add_module('net_u_linear_50', nn.Linear(20, 4))

        self.lmd_e = torch.nn.Parameter(torch.tensor([30], requires_grad=True, dtype=torch.double) )

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        output = torch.cat((u, w), dim=1)
        return output

    def loss_pde(self, x):
        y_u = self.net_u(x)
        y_w = self.net_w(x)

        ##net_u
        u, v, p, phi = y_u[:, 0], y_u[:, 1], y_u[:, 2], y_u[:, 3]
        u_g, v_g, p_g, phi_g = gradients(u, x)[0], gradients(v, x)[0], gradients(p, x)[0], gradients(phi, x)[0]

        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        v_t, v_x, v_y = v_g[:, 0], v_g[:, 1], v_g[:, 2]
        p_x, p_y = p_g[:, 1], p_g[:, 2]
        phi_t, phi_x, phi_y = phi_g[:, 0], phi_g[:, 1], phi_g[:, 2]

        u_xx, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_y, x)[0][:, 2]
        v_xx, v_yy = gradients(v_x, x)[0][:, 1], gradients(v_y, x)[0][:, 2]
        phi_xx, phi_xy, phi_yy = gradients(phi_x, x)[0][:, 1], gradients(phi_x, x)[0][:, 2], gradients(phi_y, x)[0][:, 2]

        ##net_w
        w, psi_1, psi_2 = y_w[:, 0], y_w[:, 1], y_w[:, 2]
        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:, 1], w_g[:, 2]
        w_xx, w_yy = gradients(w_x, x)[0][:, 1], gradients(w_y, x)[0][:, 2]

        ## Left Cauchy-Green Deformation Tensor
        psi_1_g, psi_2_g = gradients(psi_1, x)[0], gradients(psi_2, x)[0]
        psi_1_t, psi_1_x, psi_1_y = psi_1_g[:, 0], psi_1_g[:, 1], psi_1_g[:, 2]
        psi_2_t, psi_2_x, psi_2_y = psi_2_g[:, 0], psi_2_g[:, 1], psi_2_g[:, 2]

        B_11 = (1-phi)/2*(psi_1_x**2 + psi_2_x**2)
        B_21 = (1-phi)/2*(psi_1_x*psi_1_y + psi_2_x*psi_2_y)
        B_22 = (1-phi)/2*(psi_1_y**2 + psi_2_y**2)

        B_11_g = gradients(B_11, x)[0]
        B_21_g = gradients(B_21, x)[0]
        B_22_g = gradients(B_22, x)[0]

        B_11_x, B_21_y, B_12_x, B_22_y = B_11_g[:, 1], B_21_g[:, 2], B_21_g[:, 1], B_22_g[:, 2]

        ## NS
        ### there is another equivalent form see 5.13c
        res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) + mu*(1-phi)*u/(2*K) \
                + lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y) + self.lmd_e*(B_11_x + B_21_y)
        res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) + mu*(1-phi)*v/(2*K) \
                + lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy) + self.lmd_e*(B_12_x + B_22_y)
        res_incomp = u_x + v_y

        ## PF
        F_g = phi**3 - phi
        res_w = w + h**2*(phi_xx + phi_yy) - F_g
        res_pf = phi_t + (u*phi_x + v*phi_y) - tau*(w_xx + w_yy)/h**2
        res_psi_x = psi_1_t + (u*psi_1_x + v*psi_1_y)
        res_psi_y = psi_2_t + (u*psi_2_x + v*psi_2_y)

        # print(f'res_ns_x shape:{res_ns_x.shape}')
        # print(f'res_ns_y shape:{res_ns_y.shape}')
        # print(f'res_incomp shape:{res_incomp.shape}')
        # print(f'res_w shape:{res_w.shape}')
        # print(f'res_pf shape:{res_pf.shape}')
        # print(f'res_psi_x shape:{res_psi_x.shape}')
        # print(f'res_psi_y shape:{res_psi_y.shape}')

        print(f'res_ns_x: {(res_ns_x**2).mean()}')
        print(f'res_ns_y: {(res_ns_y**2).mean()}')
        print(f'res_psi_x: {(res_psi_x**2).mean()}')
        print(f'res_psi_y: {(res_psi_y**2).mean()}')
        print(f'res_pf: {(res_pf**2).mean()}')
        print(f'coh term: {((lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y))**2).mean()}')
        print(f'vis-elas term: {((self.lmd_e*(B_12_x + B_22_y))**2).mean()}')
        print(f'FSI term: {(mu*(1-phi)*u/(2*K)).mean()}')

        return (res_pf**2).mean() + (res_incomp**2).mean() + (res_ns_x**2).mean() + (res_ns_y**2).mean() + (res_w**2).mean() \
             + (res_psi_x**2).mean() + (res_psi_y**2).mean()

    def loss_data(self, x_data, u_data, v_data, p_data, phi_data, psi_1_data, psi_2_data):
        y_u = self.net_u(x_data)
        w_u = self.net_w(x_data)
        u, v, p, phi = y_u[:, 0], y_u[:, 1], y_u[:, 2], y_u[:, 3]
        psi_1, psi_2 = w_u[:, 1], w_u[:, 2]

        # print(f'(u - u_data)**2: {((u - u_data)**2).shape}')
        # print(f'loss_data u shape: {((u - u_data)**2).shape}')
        # print(f'loss_data v shape: {((v - v_data)**2).shape}')
        # print(f'loss_data p shape: {((p - p_data)**2).shape}')
        # print(f'loss_data phi shape: {((phi - phi_data)**2).shape}')
        # print(f'loss_data psi_1 shape: {((psi_1 - psi_1_data)**2).shape}')
        # print(f'loss_data psi_2 shape: {((psi_2 - psi_2_data)**2).shape}')

        return ((u - u_data)**2).mean() + ((v - v_data)**2).mean() + ((p - p_data)**2).mean() + \
            ((phi - phi_data)**2).mean() + ((psi_1 - psi_1_data)**2).mean() + ((psi_2 - psi_2_data)**2).mean()


    def loss_bc(self, x_l, x_r, x_up, x_dw, u_up_real):
        ## periodic left&right, low wall, up moving
        y_u_l, y_u_r, y_u_up, y_u_dw = self.net_u(x_l), self.net_u(x_r), self.net_u(x_up), self.net_u(x_dw)
        y_w_l, y_w_r, y_w_up, y_w_dw = self.net_w(x_l), self.net_w(x_r), self.net_w(x_up), self.net_w(x_dw)
        
        ## net_u
        u_l, v_l, p_l, phi_l = y_u_l[:, 0], y_u_l[:, 1], y_u_l[:, 2], y_u_l[:, 3]
        u_r, v_r, p_r, phi_r = y_u_r[:, 0], y_u_r[:, 1], y_u_r[:, 2], y_u_r[:, 3]
        u_up, v_up, phi_up = y_u_up[:, 0], y_u_up[:, 1], y_u_up[:, 3]
        u_dw, v_dw, phi_dw = y_u_dw[:, 0], y_u_dw[:, 1], y_u_dw[:, 3]
        ## net_w
        w_l, psi_1_l, psi_2_l = y_w_l[:, 0], y_w_l[:, 1], y_w_l[:, 2]
        w_r, psi_1_r, psi_2_r = y_w_r[:, 0], y_w_r[:, 1], y_w_r[:, 2]
        w_up, psi_1_up, psi_2_up = y_w_up[:, 0], y_w_up[:, 1], y_w_up[:, 2]
        w_dw, psi_1_dw, psi_2_dw = y_w_dw[:, 0], y_w_dw[:, 1], y_w_dw[:, 2]

        phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]

        w_up_y = gradients(w_up, x_up)[0][:, 2]
        w_dw_y = gradients(w_dw, x_dw)[0][:, 2]

        psi_1_l_x = gradients(psi_1_l, x_l)[0][:, 1]
        psi_2_l_x = gradients(psi_2_l, x_l)[0][:, 1]
        psi_1_r_x = gradients(psi_1_r, x_r)[0][:, 1]
        psi_2_r_x = gradients(psi_2_r, x_r)[0][:, 1]
        psi_1_up_y = gradients(psi_1_up, x_up)[0][:, 2]
        psi_2_up_y = gradients(psi_2_up, x_up)[0][:, 2]
        psi_1_dw_y = gradients(psi_1_dw, x_dw)[0][:, 2]
        psi_2_dw_y = gradients(psi_2_dw, x_dw)[0][:, 2]

        # print(f'u_up - u_up_real: {((u_up - u_up_real)**2).shape}')
        loss_up = ((u_up - u_up_real)**2).mean() + (v_up**2).mean() + (phi_up_y**2).mean() + (w_up_y**2).mean()\
                + (psi_1_up_y**2).mean() + (psi_2_up_y**2).mean()
        loss_dw = (u_dw**2).mean() + (v_dw**2).mean() + (phi_dw_y**2).mean() + (w_dw_y**2).mean()\
                + (psi_1_dw_y**2).mean() + (psi_2_dw_y**2).mean()
        loss_lr = ((u_l - u_r)**2).mean() + ((v_l - v_r)**2).mean() + ((w_l - w_r)**2).mean() + ((phi_l - phi_r)**2).mean() \
                + ((p_l - p_r)**2).mean() + (psi_1_l_x**2).mean() + (psi_2_l_x**2).mean() + (psi_1_r_x**2).mean() + (psi_2_r_x**2).mean() 
        
        print(f'loss_up: {loss_up}')
        print(f'loss_dw: {loss_dw}')
        print(f'loss_lr: {loss_lr}')

        return loss_up + loss_dw + loss_lr
                 
    def loss_ic(self, x, u_i, v_i, p_i, phi_i, psi_1_i, psi_2_i):
        y_u_pred = self.net_u(x)
        y_w_pred = self.net_w(x)
        u_i_pred, v_i_pred, p_i_pred, phi_i_pred = y_u_pred[:, 0][:,None], y_u_pred[:, 1][:,None], y_u_pred[:, 2][:,None], y_u_pred[:, 3][:,None]
        psi_1_i_pred, psi_2_i_pred = y_w_pred[:, 1][:,None], y_w_pred[:, 2][:,None]

        # print(f'u_i_pred shape: {u_i_pred.shape}')
        # print(f'u_i shape: {u_i.shape}')

        # print(f'loss_ic_u: {((u_i_pred-u_i)**2).shape}')
        # print(f'loss_ic_v: {((v_i_pred-v_i)**2).shape}')
        # print(f'loss_ic_p: {((p_i_pred-p_i)**2).shape}')
        # print(f'loss_ic_phi: {((phi_i_pred-phi_i)**2).shape}')
        # print(f'w_loss: {((w_i_pred-w_i)**2).shape}')
        # print(f'psi_1_loss: {((psi_1_i_pred-psi_1_i)**2).shape}')
        # print(f'psi_2_loss: {((psi_2_i_pred-psi_2_i)**2).shape}')
        
        return ((u_i_pred-u_i)**2).mean() + ((v_i_pred-v_i)**2).mean() + ((p_i_pred-p_i)**2).mean() \
            + ((phi_i_pred-phi_i)**2).mean() + ((psi_1_i_pred-psi_1_i)**2).mean() + ((psi_2_i_pred-psi_2_i)**2).mean()

# def CH_2D_init(x_init):
#     ## u, v
#     u_init = np.zeros((x_init.shape[0]))
#     u_init = (1-x_init[:,2])*x_init[:,2]
#     v_init = np.zeros((x_init.shape[0]))

#     #########
#     x, y = symbols('x y')
#     phi = 0.5*(tanh((x**2 + y**2-0.3**2) / (sqrt(2)*0.01) ) + tanh( (x**2 + y**2-0.7**2) / (sqrt(2)*0.01)))
#     phi_x, phi_y = phi.diff(x), phi.diff(y)
#     phi_xx, phi_yy = phi_x.diff(x), phi_y.diff(y)

#     f_phi = lambdify([x, y], phi, 'numpy')
#     f_phi_xx = lambdify([x, y], phi_xx, 'numpy')
#     f_phi_yy = lambdify([x, y], phi_yy, 'numpy')

#     phi_init = f_phi(x_init[:,1], x_init[:,2])
#     w_init = -h**2*(f_phi_xx(x_init[:,1], x_init[:,2]) + f_phi_yy(x_init[:,1],x_init[:,2])) + (phi_init**3 - phi_init)

#     return u_init, v_init, phi_init, w_init

def left_bound_vel(x):
    return 0.2*np.sin(1.2*np.pi*x[:, 0])

def bound_ext(x, t):
    num_t, num_x = t.shape[0], x.shape[0]
    x_ext, t_ext = np.tile(x, [num_t,1]), np.zeros((num_t*num_x, 1))
    for i in range (0, num_t):
        t_ext[i*num_x:(i+1)*num_x,:] = t[i]
    xt_ext = np.concatenate((t_ext, x_ext), axis=1)
    return xt_ext

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
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    epochs = 100000
    lr = 0.001
    init_path = './InitConds/'
    path_list = ['./Predictions/u/', './Predictions/v/', './Predictions/p/', './Predictions/phi/', './Predictions/w/', './Predictions/psi_1/', './Predictions/psi_2/']
    model_path = 'CheckPts/model_chkpts.pt'
    restart = 0 # restart = 0: start from beginning, restart = 1: resuming training
    check_step = 10000

    file_start = 15
    file_end = 33
    dt = 0.06

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start
    num_bh_train = 1500   # number of sampling points on verticle boundary over time
    num_bv_train = 1500   # number of sampling points on horizontal boundary over time
    num_f_train = 20000   # inner sampling points
    num_ft_train = 10000 # sampling point around thrombus
    num_i_train = 5000   # initial sampling points 

    x_min, x_max = 0, 6
    y_min, y_max = 0, 1
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    ## create new folders
    os.makedirs('InitConds', exist_ok=True)
    os.makedirs('Predictions', exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## load data file
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/K_001_lmd_e_10/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/K_001_lmd_e_10/dat_phase/chanp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        x_tmp = np.concatenate(( 0.02 + file_id*dt*np.ones((data_u.shape[0], 1)), data_u[:,0:2]), axis=1)
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:, 7:9]), axis=1)

        if file_id == file_start:
            x_data = x_tmp
            y_data = y_tmp

            ## initial condition
            xt_init = x_tmp
            u_init = y_tmp[:, 0][:, None]
            v_init = y_tmp[:, 1][:, None]
            p_init = y_tmp[:, 2][:, None]
            phi_init = y_tmp[:, 3][:, None]
            psi_1_init = y_tmp[:, 4][:, None]
            psi_2_init = y_tmp[:, 5][:, None]
        else:
            x_data = np.concatenate((x_data, x_tmp), axis=0)
            y_data = np.concatenate((y_data, y_tmp), axis=0)

    ## geometry
    x1 = np.linspace(x_min, x_max, num=num_x1)
    x2 = np.linspace(y_min, y_max, num=num_x2)
    t = np.linspace(dt*(file_start+1), dt*(file_end+1), num=num_t+1)

    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_2d = np.concatenate((x1_grid.flatten()[:,None], x2_grid.flatten()[:,None]), axis=1)

    ## initial condition
    inits = np.zeros((num_x1*num_x2, 6 ))
    inits[:,0] = u_init[:,0]
    inits[:,1] = v_init[:,0]
    inits[:,2] = p_init[:,0]
    inits[:,3] = phi_init[:,0]
    inits[:,4] = psi_1_init[:,0]
    inits[:,5] = psi_2_init[:,0]

    ## boundary condition
    x_l = np.concatenate((x1_grid[:,0][:,None],x2_grid[:,0][:,None]), axis=1)
    x_r = np.concatenate((x1_grid[:,-1][:,None], x2_grid[:,-1][:,None]), axis=1)
    x_up = np.concatenate((x1_grid[-1,:][:,None], x2_grid[-1,:][:,None]), axis=1)
    x_dw = np.concatenate((x1_grid[0,:][:,None], x2_grid[0,:][:,None]), axis=1)

    xt_l_ext = bound_ext(x_l, t)
    xt_r_ext = bound_ext(x_r, t)
    xt_up_ext = bound_ext(x_up, t)
    xt_dw_ext = bound_ext(x_dw, t)

    u_up = left_bound_vel(xt_up_ext) ## 1D array

    ## save init fig
    ## Create Folders
    for path in path_list:
        os.makedirs(path, exist_ok=True)
    os.makedirs(init_path, exist_ok=True)
    
    ## sampling
    id_f = np.random.choice(num_x1*num_x2*num_t, num_f_train, replace=False)
    id_bv = np.random.choice(num_x1*num_t, num_bv_train, replace=False)
    id_bh = np.random.choice(num_x1*num_t, num_bh_train, replace=False)
    id_i = np.random.choice(num_x1*num_x2, num_i_train, replace=False)

    ### initial
    xt_i_sample = xt_init[id_i, :]
    u_i_sample = u_init[id_i]
    v_i_sample = v_init[id_i]
    p_i_sample = p_init[id_i]
    phi_i_sample = phi_init[id_i]
    psi_1_i_sample = psi_1_init[id_i]
    psi_2_i_sample = psi_2_init[id_i]
    ### boundary
    xt_l_sample, xt_r_sample = xt_l_ext[id_bv,:], xt_r_ext[id_bv,:]
    xt_up_sample, xt_dw_sample = xt_up_ext[id_bh,:], xt_dw_ext[id_bh,:]
    u_up_sample = u_up[id_bh]
    ### inner points
    xt_f_sample = x_data[id_f,:]
    u_f_sample = y_data[id_f,0]
    v_f_sample = y_data[id_f,1]
    p_f_sample = y_data[id_f,2]
    phi_f_sample = y_data[id_f,3]
    psi_1_f_sample = y_data[id_f,4]
    psi_2_f_sample = y_data[id_f,5]


    ### sample around thrombus, concatenate
    x_thb_l = 2.75
    x_thb_r = 3.75
    x_thb_data = np.zeros((1, 3))
    y_thb_data = np.zeros((1, 6))
    for i in range(0, x_data.shape[0]):
        if x_data[i,1] >= x_thb_l and x_data[i, 1]<= x_thb_r:
            x_thb_data = np.concatenate((x_thb_data, x_data[i,:][None,:]), axis=0)
            y_thb_data = np.concatenate((y_thb_data, y_data[i,:][None,:]), axis=0)
  
    x_thb_data = np.delete(x_thb_data, 0, axis=0)
    y_thb_data = np.delete(y_thb_data, 0, axis=0)

    id_ft = np.random.choice(x_thb_data.shape[0], num_ft_train, replace=False)
    print(f'x_thb length: {x_thb_data.shape[0]}')
    x_thb_sample = x_thb_data[id_ft,:]
    u_thb_sample = y_thb_data[id_ft, 0]
    v_thb_sample = y_thb_data[id_ft, 1]
    p_thb_sample = y_thb_data[id_ft, 2]
    phi_thb_sample = y_thb_data[id_ft, 3]
    psi_1_thb_sample = y_thb_data[id_ft, 4]
    psi_2_thb_sample = y_thb_data[id_ft, 5]

    xt_f_sample = np.concatenate((xt_f_sample, x_thb_sample), axis=0)
    u_f_sample = np.concatenate((u_f_sample, u_thb_sample), axis=0)
    v_f_sample = np.concatenate((v_f_sample, v_thb_sample), axis=0)
    p_f_sample = np.concatenate((p_f_sample, p_thb_sample), axis=0)
    phi_f_sample = np.concatenate((phi_f_sample, phi_thb_sample), axis=0)
    psi_1_f_sample = np.concatenate((psi_1_f_sample, psi_1_thb_sample), axis=0)
    psi_2_f_sample = np.concatenate((psi_2_f_sample, psi_2_thb_sample), axis=0)

    ## set data as tensor and send to device
    xt_i_train = torch.tensor(xt_i_sample, requires_grad=True, dtype=torch.double).to(device)
    u_i_train = torch.tensor(u_i_sample, dtype=torch.double).to(device)
    v_i_train = torch.tensor(v_i_sample, dtype=torch.double).to(device)
    p_i_train = torch.tensor(p_i_sample, dtype=torch.double).to(device)
    phi_i_train = torch.tensor(phi_i_sample, dtype=torch.double).to(device)
    psi_1_i_train = torch.tensor(psi_1_i_sample, dtype=torch.double).to(device)
    psi_2_i_train = torch.tensor(psi_2_i_sample, dtype=torch.double).to(device)

    xt_f_train = torch.tensor(xt_f_sample, requires_grad=True, dtype=torch.double).to(device)
    u_f_train = torch.tensor(u_f_sample, dtype=torch.double).to(device)
    v_f_train = torch.tensor(v_f_sample, dtype=torch.double).to(device)
    p_f_train = torch.tensor(p_f_sample, dtype=torch.double).to(device)
    phi_f_train = torch.tensor(phi_f_sample, dtype=torch.double).to(device)
    psi_1_f_train = torch.tensor(psi_1_f_sample, dtype=torch.double).to(device)
    psi_2_f_train = torch.tensor(psi_2_f_sample, dtype=torch.double).to(device)
    
    xt_l_train = torch.tensor(xt_l_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_r_train = torch.tensor(xt_r_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_up_train = torch.tensor(xt_up_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_dw_train = torch.tensor(xt_dw_sample, requires_grad=True, dtype=torch.double).to(device)
    u_up_train = torch.tensor(u_up_sample, dtype=torch.double).to(device)


    ## instantiate model/optimizer
    model = Model().to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    ## loading model/optimizer state
    if restart == 1:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    ## training
    loss_his = np.zeros((epochs,1))
    lmd_e_his = np.zeros((epochs,1))

    def train(epoch):
        model.train()
        print(model.lmd_e)
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_data = model.loss_data(xt_f_train, u_f_train, v_f_train, p_f_train, phi_f_train, psi_1_f_train, psi_2_f_train)        
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train, u_up_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train, v_i_train, p_i_train, phi_i_train, psi_1_i_train, psi_2_i_train)
            loss = loss_pde + loss_bc + loss_data + loss_ic
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}, loss_data:{loss_data:6f}', flush=True)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        loss_his[epoch,0] = loss_value
        lmd_e_his[epoch, 0] = to_numpy(model.lmd_e)
        # print(f'epoch {epoch}: loss {loss_value:.6f}')
        
        ## checkpoint saving
        if (epoch+1)%check_step == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, model_path)

    print('start training...', flush=True)
    tic = time.time()
    for epoch in range(0, epochs):    
        train(epoch)
    toc = time.time()
    print(f'total training time: {int((toc-tic)/60)} min', flush=True)

    ## testing
    Predictions = np.zeros((7, num_t, num_x1, num_x2))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.double).to(device)
        y_pred = model(xt_tensor)
        for j in range(0, 7):
            Predictions[j, i, :, :] = to_numpy(y_pred[:, j]).reshape(num_x1, num_x2)

    for i in range(0, num_t):
        print(f'Saving Step: {i}')
        for j in range(0, 7):
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
        
    np.savetxt('loss.his', loss_his)
    np.savetxt('lmd_e.his', lmd_e_his)

if __name__ == '__main__':
    main()


















