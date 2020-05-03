"""
Author: Minglang Yin
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main(): 
    ## parameters
    file_start = 0
    file_end = 20
    dt = 0.06
    t_gap = 7

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start

    x_min, x_max = 0, 6
    y_min, y_max = 0, 2
    asp_ratio = (y_max-y_min)/(x_max-x_min)

    ## geometry
    x1 = np.linspace(x_min, x_max, num=num_x1)
    x2 = np.linspace(y_min, y_max, num=num_x2)
    t = np.linspace(0.02+dt*file_start, 0.02+dt*(file_end-1), num=num_t)

    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_2d = np.concatenate((x1_grid.flatten()[:,None], x2_grid.flatten()[:,None]), axis=1)

    ## load data file
    y_data_2 = np.zeros((6, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/Lmd_e_15_K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/Lmd_e_15_K_001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:, 7:9]), axis=1)
        for j in range (0, 6):
            y_data_2[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    y_data_10 = np.zeros((6, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/Lmd_e_01_K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/Lmd_e_01_K_001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:, 7:9]), axis=1)
        for j in range (0, 6):
            y_data_10[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    ## v and phi
    phi_2 = np.zeros((num_t, num_x1, num_x2))
    vel_2 = np.zeros((num_t, num_x1, num_x2))
    p_2 = np.zeros((num_t, num_x1, num_x2))
    psi_2_1 = np.zeros((num_t, num_x1, num_x2))
    psi_2_2 = np.zeros((num_t, num_x1, num_x2))
    
    phi_10 = np.zeros((num_t, num_x1, num_x2))
    vel_10 = np.zeros((num_t, num_x1, num_x2))
    p_10 = np.zeros((num_t, num_x1, num_x2))
    psi_10_1 = np.zeros((num_t, num_x1, num_x2))
    psi_10_2 = np.zeros((num_t, num_x1, num_x2))

    for i in range(0, num_t):
        phi_2[i, :, :] = y_data_2[3, i, :, :]
        vel_2[i, :, :] = np.sqrt(y_data_2[0, i, :, :]**2 + y_data_2[1, i, :, :]**2)
        p_2[i, :, :] = y_data_2[2, i, :, :]
        psi_2_1[i, :, :] = y_data_2[4, i, :, :]
        psi_2_2[i, :, :] = y_data_2[5, i, :, :]

        phi_10[i, :, :] = y_data_10[3, i, :, :]
        vel_10[i, :, :] = np.sqrt(y_data_10[0, i, :, :]**2 + y_data_10[1, i, :, :]**2)
        p_10[i, :, :] = y_data_10[2, i, :, :]
        psi_10_1[i, :, :] = y_data_10[4, i, :, :]
        psi_10_2[i, :, :] = y_data_10[5, i, :, :]

    ## plotting phi
    phi_range = np.linspace(-1, 1, 3)[:,None]
    phi_error = np.linspace(0, 0.1, 3)[:,None]

    fig = plt.figure(constrained_layout=False, figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 1)

    for i in range (0, 3):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(phi_2[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(phi_10[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(phi_2[t_gap*i, :, :] - phi_10[t_gap*i, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_error, cax = axins, orientation='horizontal')

    plt.savefig('phase.png')
    plt.close()

    ## plotting velocity
    vel_range = np.linspace(0, 1.0, 3)[:,None]
    vel_error = np.linspace(0, 0.1, 3)[:,None]
    fig = plt.figure(constrained_layout=False, figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 1)

    for i in range (0, 3):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(vel_2[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
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
        _h = _ax.imshow(vel_10[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
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
        _h = _ax.imshow(np.abs(vel_2[t_gap*i, :, :] - vel_10[t_gap*i, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_error, cax = axins, orientation='horizontal')

    plt.savefig('velo.png')
    plt.close()

    ## plotting pressure
    p_range = np.linspace(-25, 25, 3)[:,None]
    p_error = np.linspace(0, 1, 3)[:,None]
    fig = plt.figure(constrained_layout=False, figsize=(9, 4.5))
    gs = fig.add_gridspec(3, 1)

    for i in range (0, 3):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(p_2[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-25, vmax=25)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(p_10[t_gap*i, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-25, vmax=25)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(p_2[t_gap*i, :, :] - p_10[t_gap*i, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=p_error, cax = axins, orientation='horizontal')

    plt.savefig('pressure.png')
    plt.close()

if __name__ == '__main__':
    main()


















