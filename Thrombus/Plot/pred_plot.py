"""
Author: Minglang Yin
"""

import sys
import os
sys.path.insert(0,'../../Utils')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from plotting import newfig, savefig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib
matplotlib.use('Agg')

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
    model_path = '../Inv_K/test_3/CheckPts/model_chkpts.pt'

    file_start = 10
    file_end = 20
    dt = 0.03

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start

    x_min, x_max = 0, 6
    y_min, y_max = 0, 2
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    ## load data file
    y_data = np.zeros((4, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../Data/CH/phase_field_xiaoning/K_0001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../Data/CH/phase_field_xiaoning/K_0001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate((data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4]), axis=1)
        for j in range (0, 4):
            y_data[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    ## geometry
    x1 = np.linspace(x_min, x_max, num=num_x1)
    x2 = np.linspace(y_min, y_max, num=num_x2)
    t = np.linspace(dt*(file_start+1), dt*(file_end+1), num=num_t+1)

    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_2d = np.concatenate((x1_grid.flatten()[:,None], x2_grid.flatten()[:,None]), axis=1)

    ## instantiate model/optimizer
    model = Model().to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    ## loading model/optimizer state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## testing
    Predictions = np.zeros((5, num_t, num_x1, num_x2))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.double).to(device)
        y_pred = model(xt_tensor)
        for j in range(0, 5):
            Predictions[j, i, :, :] = to_numpy(y_pred[:, j]).reshape(num_x1, num_x2)
    
    ## v and phi
    phi_pred = np.zeros((num_t, num_x1, num_x2))
    phi_ref = np.zeros((num_t, num_x1, num_x2))
    vel_pred = np.zeros((num_t, num_x1, num_x2))
    vel_ref = np.zeros((num_t, num_x1, num_x2))
    for i in range(0, num_t):
        phi_pred[i, :, :] = Predictions[3, i, :, :]
        phi_ref[i, :, :] = y_data[3, i, :, :]
        vel_pred[i, :, :] = np.sqrt(Predictions[0, i, :, :]**2 + Predictions[1, i, :, :]**2)
        vel_ref[i, :, :] = np.sqrt(y_data[0, i, :, :]**2 + y_data[1, i, :, :]**2)

    ## plotting phi
    fig, ax = plt.subplots(3, 3, constrained_layout=True, figsize=[11, 3.5])
    gs0 = gridspec.GridSpec(3, 4)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    v_phi = np.linspace(-1, 1, 5)[:,None]
    v_error = np.linspace(0, 0.1, 5)[:,None]
    ## t0
    for i in range (0, 3):
        _ax = ax[i, 0]
        _ax.axis('off') 
        _h = _ax.imshow(phi_pred[4*i, :, :], 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        fig.colorbar(_h, ax=_ax, ticks=v_phi)
        # ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)

        _ax = ax[i, 1]
        _ax.axis('off') 
        _h = _ax.imshow(phi_pred[4*i, :, :], 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        fig.colorbar(_h, ax=_ax, ticks=v_phi)

        _ax = ax[i, 2]
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(vel_pred[i, :, :] - vel_ref[i, :, :]), 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.1)
        fig.colorbar(_h, ax=_ax, ticks=v_error)

    fig.savefig('phase.png')
    plt.close()

    ## plotting velocity
    fig, ax = plt.subplots(3,3, constrained_layout=True, figsize=[11, 3.5])
    gs0 = gridspec.GridSpec(3, 4)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    v_velocity = np.linspace(0, 0.3, 5)[:,None]
    v_error = np.linspace(0, 0.1, 5)[:,None]
    ## t0
    for i in range (0, 3):
        _ax = ax[i, 0]
        _ax.axis('off') 
        _h = _ax.imshow(vel_pred[i, :, :], 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.3)
        fig.colorbar(_h, ax=_ax, ticks=v_phi)
        # ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)

        _ax = ax[i, 1]
        _ax.axis('off') 
        _h = _ax.imshow(vel_ref[i, :, :], 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.3)
        fig.colorbar(_h, ax=_ax, ticks=v_phi)

        _ax = ax[i, 2]
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(vel_pred[4*i, :, :]-vel_ref[4*i, :, :]), 
                        interpolation='nearest', 
                        cmap='coolwarm', origin='lower', 
                        aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.1)
        fig.colorbar(_h, ax=_ax, ticks=v_error)

    fig.savefig('velocity.png')
    plt.close()

if __name__ == '__main__':
    main()


















