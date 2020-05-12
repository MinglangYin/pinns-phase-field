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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.insert(0,'../')
from Utils import *
from pinns import *

def main():
    ## parameters
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    file_start = 0
    file_end = 20
    dt = 0.03
    t_gap = 7

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    asp_ratio = (y_max-y_min)/(x_max-x_min)
    ## load data file
    y_data = np.zeros((6, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/membrane/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/membrane/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate((data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:,7:9]),axis=1)
        for j in range (0, 6):
            y_data[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    ## geometry
    x1 = np.linspace(x_min, x_max, num=num_x1)
    x2 = np.linspace(y_min, y_max, num=num_x2)
    t = np.linspace(0.03+dt*file_start, 0.03+dt*(file_end-1), num=num_t)

    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_2d = np.concatenate((x1_grid.flatten()[:,None], x2_grid.flatten()[:,None]), axis=1)

    ## instantiate model/optimizer
    model = Model().to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    ## loading model/optimizer state
    model_path = ['../../Inv_L/test_37/CheckPts/model_chkpts.pt']

    Predictions = np.zeros((4, 7, num_t, num_x1, num_x2))
    for k in range (0, 1):
        checkpoint = torch.load(model_path[k])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        ## testing
        for i in range(0, num_t):
            xt = np.concatenate((t[i]*np.ones((num_x1*num_x2, 1)), x_2d), axis=1)
            xt_tensor = torch.tensor(xt, requires_grad=True, dtype=torch.double).to(device)
            y_pred = model(xt_tensor)
            for j in range(0, 7):
                Predictions[k, j, i, :, :] = to_numpy(y_pred[:, j]).reshape(num_x1, num_x2)
    
    ## v and phi
    phi_inf = np.zeros((num_t, num_x1, num_x2))
    phi_ref = np.zeros((num_t, num_x1, num_x2))
    vel_inf = np.zeros((num_t, num_x1, num_x2))
    vel_ref = np.zeros((num_t, num_x1, num_x2))
    p_inf = np.zeros((num_t, num_x1, num_x2))
    p_ref = np.zeros((num_t, num_x1, num_x2))
    psi1_inf = np.zeros((num_t, num_x1, num_x2))
    psi1_ref = np.zeros((num_t, num_x1, num_x2))
    psi2_inf = np.zeros((num_t, num_x1, num_x2))
    psi2_ref = np.zeros((num_t, num_x1, num_x2))
    for k in range(0, 1):
        for i in range(0, num_t):
            phi_inf[i, :, :] = Predictions[k, 3, i, :, :]
            phi_ref[i, :, :] = y_data[3, i, :, :]
            vel_inf[i, :, :] = np.sqrt(Predictions[k, 0, i, :, :]**2 + Predictions[k, 1, i, :, :]**2)
            vel_ref[i, :, :] = np.sqrt(y_data[0, i, :, :]**2 + y_data[1, i, :, :]**2)
            p_inf[i, :, :] = Predictions[k, 2, i, :, :]
            p_ref[i, :, :] = y_data[2, i, :, :]
            psi1_inf[i, :, :] = Predictions[k, 5, i, :, :]
            psi1_ref[i, :, :] = y_data[4, i, :, :]
            psi2_inf[i, :, :] = Predictions[k, 6, i, :, :]
            psi2_ref[i, :, :] = y_data[5, i, :, :] 

        print(f'{k}')
        l2_error = get_relative_l2_error(vel_inf, vel_ref)
        print(f'mean relative l2 error (vel) full time: {l2_error.mean()}')
        # print(f'mean relative l2 error (vel) full time: {l2_error}')
        print(f'mean relative l2 error (vel) half time: {l2_error[:10].mean()}')

        l2_error = get_relative_l2_error(phi_inf, phi_ref)
        print(f'mean relative l2 error (phi) full time: {l2_error.mean()}')
        print(f'mean relative l2 error (phi) half time: {l2_error[:10].mean()}')

        l2_error = get_relative_l2_error(p_inf, p_ref)
        print(f'mean relative l2 error (p) full time: {l2_error.mean()}')
        print(f'mean relative l2 error (p) half time: {l2_error[:10].mean()}')

        l2_error = get_relative_l2_error(psi1_inf, psi1_ref)
        print(f'mean relative l2 error (psi1) full time: {l2_error.mean()}')
        print(f'mean relative l2 error (psi1) half time: {l2_error[:10].mean()}')

        l2_error = get_relative_l2_error(psi2_inf, psi2_ref)
        print(f'mean relative l2 error (psi2) full time: {l2_error.mean()}')
        print(f'mean relative l2 error (psi2) half time: {l2_error[:10].mean()}')

    ## figure phi
    tt = 15
    phi_min, phi_max = -1, 1
    v_min, v_max = 0, 2
    p_min, p_max = -1, 1

    phi_range = np.linspace(phi_min, phi_max, 3)[:,None]
    phi_error = np.linspace(0, 0.1*(phi_max-phi_min), 3)[:,None]
    vel_range = np.linspace(v_min, v_max, 3)[:,None]
    vel_error = np.linspace(0, 0.2*(v_max-v_min), 3)[:,None]
    p_range = np.linspace(p_min, p_max, 3)[:,None]
    p_error = np.linspace(0, 0.1*(p_max-p_min), 3)[:,None]

    fig = plt.figure(constrained_layout=False, figsize=(9, 10))
    gs = fig.add_gridspec(3, 1)
    _gs = gs[0].subgridspec(1, 3)

    ##(0, 0)
    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(phi_ref[tt, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=phi_min, vmax=phi_max)
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
    _ax.set_title('$\phi_{ref}$')

    ##(0, 1)
    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(phi_inf[tt, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=phi_min, vmax=phi_max)
    axins = inset_axes(_ax, width="100%", height="5%", loc='lower center', borderpad=-1)
    fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
    _ax.set_title('$\phi_{pred}$')

    ##(0, 2)
    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(phi_ref[tt, :, :] - phi_inf[tt, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.1*(phi_max-phi_min))
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=phi_error, cax = axins, orientation='horizontal')
    _ax.set_title('|$\phi_{ref}$ - $\phi_{pred}$|')

    ## plotting velocity
    _gs = gs[1].subgridspec(1, 3)

    ##(1, 0)
    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(vel_ref[tt, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=v_min, vmax=v_max)
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
    _ax.set_title('$v_{ref}$')

    ##(1, 1)
    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(vel_inf[tt, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=v_min, vmax=v_max)
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
    _ax.set_title('$v_{pred}$')

    ##(1, 2)
    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(vel_ref[tt, :, :] - vel_inf[tt, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.2*(v_max-v_min))
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=vel_error, cax = axins, orientation='horizontal')
    _ax.set_title('|$v_{ref}$ - $v_{pred}$|')

    ## plotting pressure
    _gs = gs[2].subgridspec(1, 3)

    ##(2, 0)
    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(p_ref[tt , :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=p_min, vmax=p_max)
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    _ax.set_title('$p_{ref}$')

    ##(2, 1)
    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(p_inf[tt , :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=p_min, vmax=p_max)
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    _ax.set_title('$p_{pred}$')

    ##(2, 2)
    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(p_ref[tt , :, :] - p_inf[tt, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=0.1*(p_max-p_min))
    axins = inset_axes(_ax,
                width="100%",
                height="5%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_error, cax = axins, orientation='horizontal')
    _ax.set_title('|$p_{ref}$ - $p_{pred}$|')

    plt.savefig('fig_mem.png')
    plt.close()

if __name__ == '__main__':
    main()

