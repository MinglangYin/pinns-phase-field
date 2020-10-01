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

from plotting import newfig, savefig
from xpinns import *

import matplotlib
matplotlib.use('Agg')

torch.manual_seed(123456)
np.random.seed(123456)

h = 0.05
lmd = 0.4*0.00010607
tau = 1e-6
rho = 1
mu = 0.1
K = 0.01
gamma = 1 #tau/lmd

def left_bound_vel(x):
    return 0.3*(2-x[:, 2])*x[:,2]

def bound_ext(x, t):
    num_t, num_x = t.shape[0], x.shape[0]
    x_ext, t_ext = np.tile(x, [num_t,1]), np.zeros((num_t*num_x, 1))
    for i in range (0, num_t):
        t_ext[i*num_x:(i+1)*num_x,:] = t[i]
    xt_ext = np.concatenate((t_ext, x_ext), axis=1)
    return xt_ext

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

    file_start = 10
    file_end = 20
    dt = 0.03

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start

    num_bh_train = 50   # number of sampling points on verticle boundary over time
    num_bv_train = 50   # number of sampling points on horizontal boundary over time
    num_f1_train = 2000   # inner sampling points
    num_f2_train = 8000   # inner sampling points
    num_pde1_train = 5000   # inner sampling points
    num_pde1_train = 20000   # inner sampling points
    num_i1_train = 1000   # initial sampling points 
    num_i2_train = 1000   # initial sampling points 
    num_intf_train = 1000

    x_min, x_max = 0, 6
    y_min, y_max = 0, 2
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    ## create new folders
    os.makedirs('InitConds', exist_ok=True)
    os.makedirs('Predictions', exist_ok=True)
    os.makedirs('CheckPts', exist_ok=True)

    ## load data file
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/K_0001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/K_0001/dat_phase/halfp' + str(file_id) + '.dat'
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
    
    #####
    ## subdomains
    # Boundary points
    rad_x = np.sqrt((xt_dw_ext[:, 1] - 3)**2) - 1.4
    ind_1 = rad_x > 0
    ind_2 = rad_x < 0
    xt_dw1_ext = xt_dw_ext[ind_1]
    xt_dw2_ext = xt_dw_ext[ind_2]
    # initial points
    rad_x = np.sqrt((xt_init[:, 1] - 3)**2 + xt_init[:, 2]**2) - 1.4
    ind_1 = rad_x > 0
    ind_2 = rad_x < 0
    xt_init1 = xt_init[ind_1]
    xt_init2 = xt_init[ind_2]
    inits1 = inits[ind_1]
    inits2 = inits[ind_2]
    # PDE/Data points
    rad_x = np.sqrt((x_data[:, 1] - 3)**2 + x_data[:, 2]**2) - 1.4
    ind_1 = rad_x > 0
    ind_2 = rad_x < 0
    x1_data = x_data[ind_1]
    x2_data = x_data[ind_2]
    y1_data = y_data[ind_1]
    y2_data = y_data[ind_2]
    # interface points
    num_intf = 100
    theta = np.linspace(0, np.pi, num_intf)
    x_intf = np.zeros((num_intf, 2))
    R = 1.4
    for i in range(0, num_intf):
        x_intf[i, 0] = 3 - R*np.cos(theta[i])
        x_intf[i, 1] = R*np.sin(theta[i])
    x_intf_ext = bound_ext(x_l, t)

    #####
    ## sampling
    id_data1 = np.random.choice(25400, num_f1_train, replace=False)
    id_data2 = np.random.choice(74600, num_f2_train, replace=False)

    id_bv = np.random.choice(num_x1*num_t, num_bv_train, replace=False)
    
    id_bh = np.random.choice(num_x2*num_t, num_bh_train, replace=False)
    id_bh1 = np.random.choice(460, num_bh_train, replace=False)
    id_bh2 = np.random.choice(540, num_bh_train, replace=False)

    id_i1 = np.random.choice(2540, num_i1_train, replace=False)
    id_i2 = np.random.choice(7460, num_i2_train, replace=False)

    id_pde1 = np.random.choice(2540, num_pde_train, replace=False)
    id_pde2 = np.random.choice(7460, num_pde_train, replace=False)

    id_intf = np.random.choice(num_intf*num_t, num_intf_train, replace=False)

    ### PDE
    xt_pde1_sample = x1_data[id_pde1, :]
    xt_pde2_sample = x2_data[id_pde2, :]
    ### initial
    xt_i1_sample = xt_init1[id_i1, :]
    xt_i2_sample = xt_init2[id_i2, :]
    u_i1_sample, v_i1_sample, p_i1_sample, phi_i1_sample = inits1[id_i1, 0], inits1[id_i1, 1], inits1[id_i1, 2], inits1[id_i1, 3]
    u_i2_sample, v_i2_sample, p_i2_sample, phi_i2_sample = inits2[id_i2, 0], inits2[id_i2, 1], inits2[id_i2, 2], inits2[id_i2, 3]

    ### boundary
    u_l_sample = u_l[id_bv]
    xt_l_sample, xt_r_sample = xt_l_ext[id_bv,:], xt_r_ext[id_bv,:]
    xt_up_sample  = xt_up_ext[id_bh,:]

    xt_dw1_sample = xt_dw1_ext[id_bh1,:]
    xt_dw2_sample = xt_dw2_ext[id_bh2,:]
    ### data points
    xt_f1_sample = x1_data[id_f,:]
    xt_f2_sample = x2_data[id_f,:]

    u_f1_sample, v_f1_sample, p_f1_sample, phi_f1_sample = y1_data[id_f,0], y1_data[id_f,1], y1_data[id_f,2], y1_data[id_f,3]
    u_f2_sample, v_f2_sample, p_f2_sample, phi_f2_sample = y2_data[id_f,0], y2_data[id_f,1], y2_data[id_f,2], y2_data[id_f,3]

    ## set data as tensor and send to device
    xt_pde1_train = torch.tensor(xt_pde1_sample, requires_grad=True, dtype=torch.float).to(device)
    xt_i1_train = torch.tensor(xt_i1_sample, requires_grad=True, dtype=torch.float).to(device)
    u_i1_train = torch.tensor(u_i1_sample, dtype=torch.float).to(device)
    v_i1_train = torch.tensor(v_i1_sample, dtype=torch.float).to(device)
    p_i1_train = torch.tensor(p_i1_sample, dtype=torch.float).to(device)
    phi_i1_train = torch.tensor(phi_i1_sample, dtype=torch.float).to(device)

    xt_f1_train = torch.tensor(xt_f1_sample, requires_grad=True, dtype=torch.float).to(device)
    phi_f1_train = torch.tensor(phi_f1_sample, dtype=torch.float).to(device)
    
    xt_l1_train = torch.tensor(xt_l1_sample, requires_grad=True, dtype=torch.float).to(device)
    xt_r1_train = torch.tensor(xt_r1_sample, requires_grad=True, dtype=torch.float).to(device)
    xt_up1_train = torch.tensor(xt_up1_sample, requires_grad=True, dtype=torch.float).to(device)
    xt_dw1_train = torch.tensor(xt_dw1_sample, requires_grad=True, dtype=torch.float).to(device)
    ul_train = torch.tensor(u_l1_sample, dtype=torch.float).to(device)

    ## network 2
    xt_pde2_train = torch.tensor(xt_pde2_sample, requires_grad=True, dtype=torch.float).to(device)
    xt_i2_train = torch.tensor(xt_i2_sample, requires_grad=True, dtype=torch.float).to(device)
    u_i2_train = torch.tensor(u_i2_sample, dtype=torch.float).to(device)
    v_i2_train = torch.tensor(v_i2_sample, dtype=torch.float).to(device)
    p_i2_train = torch.tensor(p_i2_sample, dtype=torch.float).to(device)
    phi_i2_train = torch.tensor(phi_i2_sample, dtype=torch.float).to(device)

    xt_f2_train = torch.tensor(xt_f2_sample, requires_grad=True, dtype=torch.float).to(device)
    phi_f2_train = torch.tensor(phi_f2_sample, dtype=torch.float).to(device)
    xt_dw1_train = torch.tensor(xt_dw2_sample, requires_grad=True, dtype=torch.float).to(device)
    
    ## instantiate model/optimizer
    model = XPINNs().to(device)
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
            loss_pde = model.loss_pde(xt_pde1_train, xt_pde2_train)
            loss_data = model.loss_data(xt_f1_train, phi_f1_train, xt_f2_train, phi_f2_train)        
            loss_bc = model.loss_bc(xt_l1_train, xt_r1_train, xt_up1_train, xt_dw1_train, ul1_train, xt_dw2_train)
            loss_ic = model.loss_ic(xt_i1_train, u_i1_train, v_i1_train, p_i1_train, phi_i1_train,
                                    xt_i2_train, u_i2_train, v_i2_train, p_i2_train, phi_i2_train)
            loss_intf = model.loss_ic(xt_i1_train)
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

    np.savetxt('loss.his', loss_his)
    np.savetxt('K.his', K_his)

    ## Ploting
    Predictions = np.zeros((5, num_t, num_x1, num_x2))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.float).to(device)
        y_pred = model(xt_tensor)
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
    

if __name__ == '__main__':
    main()


















