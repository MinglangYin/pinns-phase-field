"""
Author: Minglang Yin
"""
import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

h = 0.04
lmd = 0.4*0.00010607
#lmd_e = 0.5
tau = 1e-6
rho = 1
mu = 0.1
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

        self.lmd_e = torch.nn.Parameter(torch.tensor([15], requires_grad=True, dtype=torch.float32) )
        self.K = torch.nn.Parameter(torch.tensor([10], requires_grad=True, dtype=torch.float32) )

    def forward(self, x):
        w = self.net_w(x)
        u = self.net_u(x)
        output = torch.cat((u, w), dim=1)
        return output

## get_relative_l2_error
def get_relative_l2_error(data_inf, data_ref):
    ## data_inf and data_ref have the same size (num_t, num_x, num_y)
    [num_t, num_x, num_y] = data_inf.shape
    error = np.zeros(num_t)
    var = data_ref.var()
    for t in range(0, num_t):
        err = ((data_inf[t] - data_ref[t])**2).mean()
        error[t] = err/var
    return error


## get_relative_l2_error
def get_l2_error(data_inf, data_ref):
    ## data_inf and data_ref have the same size (num_t, num_x, num_y)
    [num_t, num_x, num_y] = data_inf.shape
    error = np.zeros(num_t)
    for t in range(0, num_t):
        error[t] = ((data_inf[t] - data_ref[t])**2).mean()

    return error

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
    model_path = '../../Inv_L/test_9/CheckPts/model_chkpts.pt'

    file_start = 0
    file_end = 20
    dt = 0.06
    t_gap = 6

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
        name_u = '../../../Data/CH/phase_field_xiaoning/Lmd_e_10_K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/Lmd_e_10_K_001/dat_phase/halfp' + str(file_id) + '.dat'
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
    t = np.linspace(0.02+dt*file_start, 0.02+dt*(file_end-1), num=num_t)

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
    Predictions = np.zeros((7, num_t, num_x1, num_x2))
    for i in range(0, num_t):
        xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
        xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.double).to(device)
        y_pred = model(xt_tensor)
        for j in range(0, 7):
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


    l2_error = get_relative_l2_error(vel_inf, vel_ref)
    print(f'mean relative l2 error (vel): {l2_error}')
    l2_error = get_relative_l2_error(vel_inf, vel_ref)
    print(f'mean relative l2 error (vel) full time: {l2_error.mean()}')
    l2_error = get_relative_l2_error(vel_inf, vel_ref)
    print(f'mean relative l2 error (vel) partial time: {l2_error[:10].mean()}')

    l2_error = get_relative_l2_error(phi_inf, phi_ref)
    print(f'mean relative l2 error (phi): {l2_error}')
    l2_error = get_relative_l2_error(phi_inf, phi_ref)
    print(f'mean relative l2 error (phi) full time: {l2_error.mean()}')
    l2_error = get_relative_l2_error(phi_inf, phi_ref)
    print(f'mean relative l2 error (phi) partial time: {l2_error[:10].mean()}')


    ## plotting phi
    # fig, ax = plt.subplots(3, 3, constrained_layout=True, figsize=[9, 4.5])
    # gs0 = gridspec.GridSpec(3, 4)
    # gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    phi_range = np.linspace(-1, 1, 3)[:,None]
    phi_error = np.linspace(0, 0.15, 4)[:,None]

    fig = plt.figure(constrained_layout=False, figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 1)

    for i in range (0, 3):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(phi_ref[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
        # if i == 0:
            # _ax.set_title('$\phi_{ref}$')


        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(phi_inf[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
        # if i == 0:
            # _ax.set_title('$\phi_{inf}$')


        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(phi_ref[t_gap*i, :, :] - phi_inf[t_gap*i, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.15)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_error, cax = axins, orientation='horizontal')
        # if i == 0:
            # _ax.set_title('$\epsilon_{\phi}$')

    # plt.gcf().text(0.05, 0.9, "(a)", fontsize=14 )

    plt.savefig('phase.png')
    plt.close()

    ## plotting velocity
    vel_range = np.linspace(0, 1.0, 3)[:,None]
    vel_error = np.linspace(0, 0.15, 4)[:,None]
    fig = plt.figure(constrained_layout=False, figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 1)

    for i in range (0, 3):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(vel_ref[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=1.0)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(vel_inf[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=1.0)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(vel_ref[t_gap*i, :, :] - vel_inf[t_gap*i, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.15)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_error, cax = axins, orientation='horizontal')

    plt.savefig('velo.png')
    plt.close()


if __name__ == '__main__':
    main()

