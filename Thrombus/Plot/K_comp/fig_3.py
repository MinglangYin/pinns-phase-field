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
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

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
    model_path = '../Inv_K/test_5/CheckPts/model_chkpts.pt'

    file_start = 0
    file_end = 20
    dt = 0.03
    t_gap = 9
    t_snap = 10

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
        name_u = '../../Data/CH/phase_field_xiaoning/K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../Data/CH/phase_field_xiaoning/K_001/dat_phase/halfp' + str(file_id) + '.dat'
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
    phi_inf = np.zeros((num_t, num_x1, num_x2))
    phi_ref = np.zeros((num_t, num_x1, num_x2))
    vel_inf = np.zeros((num_t, num_x1, num_x2))
    vel_ref = np.zeros((num_t, num_x1, num_x2))
    for i in range(0, num_t):
        phi_inf[i, :, :] = Predictions[3, i, :, :]
        phi_ref[i, :, :] = y_data[3, i, :, :]
        vel_inf[i, :, :] = np.sqrt(Predictions[0, i, :, :]**2 + Predictions[1, i, :, :]**2)
        vel_ref[i, :, :] = np.sqrt(y_data[0, i, :, :]**2 + y_data[1, i, :, :]**2)

    ## plotting phi
    # fig, ax = plt.subplots(3, 3, constrained_layout=True, figsize=[9, 4.5])
    # gs0 = gridspec.GridSpec(3, 4)
    # gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    phi_range = np.linspace(-1, 1, 3)[:,None]
    phi_error = np.linspace(0, 0.1, 3)[:,None]

    fig = plt.figure(constrained_layout=False, figsize=(12, 4))
    gs = fig.add_gridspec(2, 1)

    _gs = gs[0].subgridspec(1, 3)
    ##ax1
    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(phi_ref[t_snap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=-1, vmax=1)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
    _ax.set_title('$\phi_{ref}$', fontsize=18)

    ##ax2
    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(phi_inf[t_snap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=-1, vmax=1)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
    _ax.set_title('$\phi_{pred}$', fontsize=18)

    ##ax3
    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(phi_ref[t_snap, :, :] - phi_inf[t_snap, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.1)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=phi_error, cax = axins, orientation='horizontal')
    _ax.set_title('$\epsilon_{\phi}$', fontsize=18)

    
    ## plotting velocity
    vel_range = np.linspace(0, 0.4, 3)[:,None]
    vel_error = np.linspace(0, 0.04, 3)[:,None]

    _gs = gs[1].subgridspec(1, 3)

    ##ax4
    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(vel_ref[t_snap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.4)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
    _ax.set_title('$v_{ref}$', fontsize=18)

    ##ax5
    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(vel_inf[t_snap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.4)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
    _ax.set_title('$v_{pred}$', fontsize=18)

    ##ax6
    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(vel_ref[t_snap, :, :] - vel_inf[t_snap, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.04)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_error, cax = axins, orientation='horizontal')
    _ax.set_title('$\epsilon_{v}$', fontsize=18)

    plt.savefig('fig_3.png')
    plt.close()

if __name__ == '__main__':
    main()


















