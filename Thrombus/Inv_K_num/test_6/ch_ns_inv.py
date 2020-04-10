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
        Output: [u, v, p, phi], [w]
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

from sympy import *
from plotting import newfig, savefig

import matplotlib
matplotlib.use('Agg')


torch.manual_seed(123456)
np.random.seed(123456)

h = 0.05
lmd = 0.4*0.00010607
tau = 1e-6
rho = 1
mu = 0.1
# K = 0.01

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net_w = nn.Sequential()
        self.net_w.add_module('net_w_linear_1', nn.Linear(3, 20))
        self.net_w.add_module('net_w_tanh_1', nn.Tanh())
        for num in range(2,10):
            self.net_w.add_module('net_w_linear_%d' %(num), nn.Linear(20, 20))
            self.net_w.add_module('net_w_tanh_%d' %(num), nn.Tanh())
        self.net_w.add_module('net_w_linear_50', nn.Linear(20, 1))

        self.net_u = nn.Sequential()
        self.net_u.add_module('net_u_linear_1', nn.Linear(3, 20))
        self.net_u.add_module('net_u_tanh_1', nn.Tanh())
        for num in range(2,10):
            self.net_u.add_module('net_u_linear_%d' %(num), nn.Linear(20, 20))
            self.net_u.add_module('net_u_tanh_%d' %(num), nn.Tanh())
        self.net_u.add_module('net_u_linear_50', nn.Linear(20, 4))

        self.K = torch.nn.Parameter(torch.tensor([10], requires_grad=True, dtype=torch.double) )

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        output = torch.cat((u, w), dim=1)
        return output

    def loss_pde(self, x):
        y = self.net_u(x)
        w = self.net_w(x)[:,0]

        u, v, p, phi = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        u_g, v_g, p_g, phi_g = gradients(u, x)[0], gradients(v, x)[0], gradients(p, x)[0], gradients(phi, x)[0]
        u_t, u_x, u_y = u_g[:, 0], u_g[:, 1], u_g[:, 2]
        v_t, v_x, v_y = v_g[:, 0], v_g[:, 1], v_g[:, 2]
        p_x, p_y = p_g[:, 1], p_g[:, 2]
        phi_t, phi_x, phi_y = phi_g[:, 0], phi_g[:, 1], phi_g[:, 2]

        u_xx, u_yy = gradients(u_x, x)[0][:, 1], gradients(u_y, x)[0][:, 2]
        v_xx, v_yy = gradients(v_x, x)[0][:, 1], gradients(v_y, x)[0][:, 2]
        phi_xx, phi_xy, phi_yy = gradients(phi_x, x)[0][:, 1], gradients(phi_x, x)[0][:, 2], gradients(phi_y, x)[0][:, 2]

        w_g = gradients(w, x)[0]
        w_x, w_y = w_g[:, 1], w_g[:, 2]
        w_xx, w_yy = gradients(w_x, x)[0][:, 1], gradients(w_y, x)[0][:, 2]

        ## NS
        res_ns_x = rho*(u_t + u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy) + \
                lmd*(2*phi_x*phi_xx + phi_yy*phi_x + phi_xy*phi_y) + mu*(1-phi)*u/(2*self.K)
        res_ns_y = rho*(v_t + u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy) + \
                lmd*(phi_xy*phi_x + phi_xx*phi_y + 2*phi_y*phi_yy) + mu*(1-phi)*v/(2*self.K)
        res_incomp = u_x + v_y

        ## PF
        F_g = phi**3 - phi
        res_w = w + h**2*(phi_xx + phi_yy) - F_g  
        # res_pf = phi_t + h**2*(u*phi_x + v*phi_y) - tau*(w_xx + w_yy)
        res_pf = phi_t + (u*phi_x + v*phi_y) - tau*(w_xx + w_yy)/h**2

        # print(f'res_ns_x shape:{res_ns_x.shape}')
        # print(f'res_ns_y shape:{res_ns_y.shape}')
        # print(f'res_incomp shape:{res_incomp.shape}')
        # print(f'res_w shape:{res_w.shape}')
        # print(f'res_pf shape:{res_pf.shape}')

        return (res_pf**2).mean() + (res_incomp**2).mean() + (res_ns_x**2).mean() + (res_ns_y**2).mean() + (res_w**2).mean()

    def loss_data(self, x_data, phi_data):
        y = self.net_u(x_data)
        phi = y[:, 3]

        # print(f'loss_data u shape: {((u - u_data)**2).shape}')
        # print(f'loss_data v shape: {((v - v_data)**2).shape}')
        # print(f'loss_data p shape: {((p - p_data)**2).shape}')
        # print(f'loss_data phi shape: {((phi - phi_data)**2).shape}')

        return ((phi - phi_data)**2).mean()

    def loss_bc(self, x_l, x_r, x_up, x_dw, u_l_real):
        ## velocity inflow
        y_l, y_r, y_up, y_dw = self.net_u(x_l), self.net_u(x_r), self.net_u(x_up), self.net_u(x_dw)
        w_l, w_r, w_up, w_dw = self.net_w(x_l)[:,0], self.net_w(x_r)[:,0], self.net_w(x_up)[:,0], self.net_w(x_dw)[:,0]
        
        u_l, v_l, phi_l = y_l[:, 0], y_l[:, 1], y_l[:, 3]
        u_r, p_r, phi_r = y_r[:, 0], y_r[:, 2], y_r[:, 3]
        u_up, v_up, phi_up = y_up[:, 0], y_up[:, 1], y_up[:, 3]
        u_dw, v_dw, phi_dw = y_dw[:, 0], y_dw[:, 1], y_dw[:, 3]

        u_r_x = gradients(u_r, x_r)[0][:, 1]

        phi_l_x = gradients(phi_l, x_l)[0][:, 1]
        phi_r_x = gradients(phi_r, x_r)[0][:, 1]
        phi_up_y = gradients(phi_up, x_up)[0][:, 2]
        phi_dw_y = gradients(phi_dw, x_dw)[0][:, 2]

        w_l_x = gradients(w_l, x_l)[0][:, 1]
        w_r_x = gradients(w_r, x_r)[0][:, 1]
        w_up_y = gradients(w_up, x_up)[0][:, 2]
        w_dw_y = gradients(w_dw, x_dw)[0][:, 2]

        # print(f'u_l - u_l_real: {((u_l - u_l_real)**2).shape}')

        return ((u_l - u_l_real)**2).mean() + (u_r_x**2).mean() + (u_up**2).mean() + (u_dw**2).mean()\
                + (v_l**2).mean() + (v_up**2).mean() + (v_dw**2).mean() + (p_r**2).mean()\
                + (phi_l_x**2).mean() + (phi_r_x**2).mean() +  (phi_up_y**2).mean() + (phi_dw_y**2).mean() \
                + (w_l_x**2).mean() + (w_r_x**2).mean() + (w_up_y**2).mean() + (w_dw_y**2).mean()

    def loss_ic(self, x, u_i, v_i, p_i, phi_i):
        y_pred = self.net_u(x)
        u_i_pred, v_i_pred, p_i_pred, phi_i_pred = y_pred[:, 0][:,None], y_pred[:, 1][:,None], y_pred[:, 2][:,None], y_pred[:, 3][:,None]

        # print(f'u_i_pred shape: {u_i_pred.shape}')
        # print(f'u_i shape: {u_i.shape}')

        # print(f'loss_ic_u: {((u_i_pred-u_i)**2).shape}')
        # print(f'loss_ic_v: {((v_i_pred-v_i)**2).shape}')
        # print(f'loss_ic_p: {((p_i_pred-p_i)**2).shape}')
        # print(f'loss_ic_phi: {((phi_i_pred-phi_i)**2).shape}')
        # print(f'w_loss: {((w_i_pred-w_i)**2).shape}')
        
        return ((u_i_pred-u_i)**2).mean() + ((v_i_pred-v_i)**2).mean() + ((p_i_pred-p_i)**2).mean() + ((phi_i_pred-phi_i)**2).mean() 

def left_bound_vel(x):
    return 0.3*(2-x[:, 2])*x[:,2]

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
    epochs = 300000
    lr = 0.001
    init_path = './InitConds/'
    path_list = ['./Predictions/u/', './Predictions/v/', './Predictions/p/', './Predictions/phi/', './Predictions/w/']
    model_path = 'CheckPts/model_chkpts.pt'
    restart = 0 # restart = 0: start from beginning, restart = 1: resuming training
    check_step = 10000

    file_start = 0
    file_end = 20
    dt = 0.03

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start
    num_bh_train = 50   # number of sampling points on verticle boundary over time
    num_bv_train = 50   # number of sampling points on horizontal boundary over time
    num_f_train = 100   # inner sampling points
    num_i_train = 5000   # initial sampling points 

    x_min, x_max = 0, 6
    y_min, y_max = 0, 2
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    ## create new folders
    os.makedirs('InitConds', exist_ok=True)
    os.makedirs('Predictions', exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)
    os.makedirs('Omega', exist_ok=True)

    ## load data file
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/K_001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        x_tmp = np.concatenate(( file_id*dt*np.ones((data_u.shape[0], 1)), data_u[:,0:2]), axis=1)
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4]), axis=1)

        if file_id == file_start:
            x_data = x_tmp
            y_data = y_tmp

            ## initial condition
            xt_init = x_tmp
            u_init = y_tmp[:, 0][:, None]
            v_init = y_tmp[:, 1][:, None]
            p_init = y_tmp[:, 2][:, None]
            phi_init = y_tmp[:, 3][:, None]
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
    inits = np.zeros((num_x1*num_x2, 4 ))

    inits[:,0] = u_init[:,0]
    inits[:,1] = v_init[:,0]
    inits[:,2] = p_init[:,0]
    inits[:,3] = phi_init[:,0]
    
    ## boundary condition
    x_l = np.concatenate((x1_grid[:,0][:,None],x2_grid[:,0][:,None]), axis=1)
    x_r = np.concatenate((x1_grid[:,-1][:,None], x2_grid[:,-1][:,None]), axis=1)
    x_up = np.concatenate((x1_grid[-1,:][:,None], x2_grid[-1,:][:,None]), axis=1)
    x_dw = np.concatenate((x1_grid[0,:][:,None], x2_grid[0,:][:,None]), axis=1)

    xt_l_ext = bound_ext(x_l, t)
    xt_r_ext = bound_ext(x_r, t)
    xt_up_ext = bound_ext(x_up, t)
    xt_dw_ext = bound_ext(x_dw, t)

    u_l = left_bound_vel(xt_l_ext) ## 1D array

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
    u_i_sample, v_i_sample, p_i_sample, phi_i_sample = u_init[id_i], v_init[id_i], p_init[id_i], phi_init[id_i]
    ### boundary
    u_l_sample = u_l[id_bv]
    xt_l_sample, xt_r_sample = xt_l_ext[id_bv,:], xt_r_ext[id_bv,:]
    xt_up_sample, xt_dw_sample = xt_up_ext[id_bh,:], xt_dw_ext[id_bh,:]
    ### inner points
    xt_f_sample = x_data[id_f,:]
    u_f_sample, v_f_sample, p_f_sample, phi_f_sample = y_data[id_f,0], y_data[id_f,1], y_data[id_f,2], y_data[id_f,3]

    ## set data as tensor and send to device
    xt_i_train = torch.tensor(xt_i_sample, requires_grad=True, dtype=torch.double).to(device)
    u_i_train = torch.tensor(u_i_sample, dtype=torch.double).to(device)
    v_i_train = torch.tensor(v_i_sample, dtype=torch.double).to(device)
    p_i_train = torch.tensor(p_i_sample, dtype=torch.double).to(device)
    phi_i_train = torch.tensor(phi_i_sample, dtype=torch.double).to(device)

    xt_f_train = torch.tensor(xt_f_sample, requires_grad=True, dtype=torch.double).to(device)
    phi_f_train = torch.tensor(phi_f_sample, dtype=torch.double).to(device)
    
    xt_l_train = torch.tensor(xt_l_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_r_train = torch.tensor(xt_r_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_up_train = torch.tensor(xt_up_sample, requires_grad=True, dtype=torch.double).to(device)
    xt_dw_train = torch.tensor(xt_dw_sample, requires_grad=True, dtype=torch.double).to(device)
    ul_train = torch.tensor(u_l_sample, dtype=torch.double).to(device)
    
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
    K_his = np.zeros((epochs,1))

    def train(epoch):
        model.train()
        print(model.K)
        def closure():
            optimizer.zero_grad()
            loss_pde = model.loss_pde(xt_f_train)
            loss_data = model.loss_data(xt_f_train, phi_f_train)        
            loss_bc = model.loss_bc(xt_l_train, xt_r_train, xt_up_train, xt_dw_train, ul_train)
            loss_ic = model.loss_ic(xt_i_train, u_i_train, v_i_train, p_i_train, phi_i_train)
            loss = loss_pde + loss_bc + loss_data + 100*loss_ic
            print(f'epoch {epoch} loss_pde:{loss_pde:6f}, loss_bc:{loss_bc:6f}, loss_ic:{loss_ic:6f}, loss_data:{loss_data:6f}', flush=True)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        loss_his[epoch,0] = loss_value
        K_his[epoch, 0] = to_numpy(model.K)
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
    Predictions = np.zeros((5, num_t, num_x1, num_x2))
    Omega_pred = np.zeros((num_t, num_x1*num_x2, 8))
    
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.double).to(device)
        y_pred = model(xt_tensor)

        ## save data points
        print(f'Saving omega step: {i}')
        Omega_pred[i,:,0:3] = to_numpy(xt_tensor)
        Omega_pred[i,:,3:] = to_numpy(y_pred)
        tmp_name = './Omega/Omega_' + str(i) + '.dat'
        np.savetxt(tmp_name, Omega_pred[i])

        for j in range(0, 5):
            Predictions[j, i, :, :] = to_numpy(y_pred[:, j]).reshape(num_x1, num_x2)

    for i in range(0, num_t):
        print(f'Saving Step: {i}')
        for j in range(0, 5):
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
    np.savetxt('K.his', K_his)


if __name__ == '__main__':
    main()



