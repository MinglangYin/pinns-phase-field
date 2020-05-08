"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0,'../')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Utils import *

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
    y_data_15 = np.zeros((6, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/Lmd_e_15_K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/Lmd_e_15_K_001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:, 7:9]), axis=1)
        for j in range (0, 6):
            y_data_15[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    y_data_01 = np.zeros((6, file_end-file_start, num_x1, num_x2))
    t_step = 0
    for file_id in range(file_start, file_end):
        name_u = '../../../Data/CH/phase_field_xiaoning/Lmd_e_01_K_001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../../Data/CH/phase_field_xiaoning/Lmd_e_01_K_001/dat_phase/halfp' + str(file_id) + '.dat'
        data_u = np.loadtxt(name_u)
        data_phi = np.loadtxt(name_phi)

        ## x_tmp: [t,x,y]   u_tmp:[u,v,p,phi]
        y_tmp = np.concatenate(( data_u[:,3:5], data_u[:,6:7], data_phi[:,3:4], data_u[:, 7:9]), axis=1)
        for j in range (0, 6):
            y_data_01[j, t_step, :, :] = y_tmp[:,j].reshape(100, 100)
        t_step = t_step + 1

    ## v and phi
    phi_15 = np.zeros((num_t, num_x1, num_x2))
    vel_15 = np.zeros((num_t, num_x1, num_x2))
    p_15 = np.zeros((num_t, num_x1, num_x2))
    psi_15_1 = np.zeros((num_t, num_x1, num_x2))
    psi_15_2 = np.zeros((num_t, num_x1, num_x2))
    
    phi_01 = np.zeros((num_t, num_x1, num_x2))
    vel_01 = np.zeros((num_t, num_x1, num_x2))
    p_01 = np.zeros((num_t, num_x1, num_x2))
    psi_01_1 = np.zeros((num_t, num_x1, num_x2))
    psi_01_2 = np.zeros((num_t, num_x1, num_x2))

    for i in range(0, num_t):
        phi_15[i, :, :] = y_data_15[3, i, :, :]
        vel_15[i, :, :] = np.sqrt(y_data_15[0, i, :, :]**2 + y_data_15[1, i, :, :]**2)
        p_15[i, :, :] = y_data_15[2, i, :, :]
        psi_15_1[i, :, :] = y_data_15[4, i, :, :]
        psi_15_2[i, :, :] = y_data_15[5, i, :, :]

        phi_01[i, :, :] = y_data_01[3, i, :, :]
        vel_01[i, :, :] = np.sqrt(y_data_01[0, i, :, :]**2 + y_data_01[1, i, :, :]**2)
        p_01[i, :, :] = y_data_01[2, i, :, :]
        psi_01_1[i, :, :] = y_data_01[4, i, :, :]
        psi_01_2[i, :, :] = y_data_01[5, i, :, :]

    ## plotting phi
    phi_range = np.linspace(-1, 1, 3)[:,None]
    phi_error = np.linspace(0, 0.1, 3)[:,None]

    fig = plt.figure(constrained_layout=False, figsize=(9, 10))
    gs = fig.add_gridspec(6, 1)

    for i in range (0, 2):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(phi_15[t_gap*(i+1), :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')
        # _ax.set_title('$\phi_{ref}$')

        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(phi_01[t_gap*(i+1), :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=-1, vmax=1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        # _ax.set_title('$\phi_{pred}$')
        
        fig.colorbar(_h, ax=_ax, ticks=phi_range, cax = axins, orientation='horizontal')

        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(phi_15[t_gap*(i+1), :, :] - phi_01[t_gap*(i+1), :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.1)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=phi_error, cax = axins, orientation='horizontal')
        # _ax.set_title('|$\phi_{ref}$ - $\phi_{pred}$|')

    ## plotting velocity
    vel_range = np.linspace(0, 1.0, 3)[:,None]
    vel_error = np.linspace(0, 0.2, 3)[:,None]

    for i in range (2, 4):
        _gs = gs[i].subgridspec(1, 3)

        _ax = fig.add_subplot(_gs[0])
        _ax.axis('off') 
        _h = _ax.imshow(vel_15[t_gap*(i-1), :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=1.0)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
        # _ax.set_title('$v_{ref}$')

        _ax = fig.add_subplot(_gs[1])
        _ax.axis('off') 
        _h = _ax.imshow(vel_01[t_gap*(i-1), :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=1.0)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_range, cax = axins, orientation='horizontal')
        # _ax.set_title('$v_{pred}$')

        _ax = fig.add_subplot(_gs[2])
        _ax.axis('off') 
        _h = _ax.imshow(np.abs(vel_15[t_gap*(i-1), :, :] - vel_01[t_gap*(i-1), :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
        _h.set_clim(vmin=0, vmax=0.2)
        axins = inset_axes(_ax,
                    width="100%",
                    height="10%",
                    loc='lower center',
                    borderpad=-1,
                    )
        fig.colorbar(_h, ax=_ax, ticks=vel_error, cax = axins, orientation='horizontal')
        # _ax.set_title('|$v_{ref}$ - $v_{pred}$|')

    ## plotting pressure
    p_range = np.linspace(-20, 0, 3)[:,None]
    p_error = np.linspace(0, 1, 3)[:,None]

    _gs = gs[4].subgridspec(1, 3)

    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(p_15[t_gap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=-20, vmax=0)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    # _ax.set_title('$p_{ref}$')

    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(p_01[t_gap, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=-20, vmax=0)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    # _ax.set_title('$p_{pred}$')

    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(p_15[t_gap, :, :] - p_01[t_gap, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=1)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_error, cax = axins, orientation='horizontal')
    # _ax.set_title('|$p_{ref}$ - $p_{pred}$|')

    #######
    p_range = np.linspace(0, 10, 3)[:,None]
    p_error = np.linspace(0, 1, 3)[:,None]
    _gs = gs[5].subgridspec(1, 3)

    _ax = fig.add_subplot(_gs[0])
    _ax.axis('off') 
    _h = _ax.imshow(p_15[t_gap*2, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=10)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    # _ax.set_title('$p_{ref}$')

    _ax = fig.add_subplot(_gs[1])
    _ax.axis('off') 
    _h = _ax.imshow(p_01[t_gap*2, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=10)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_range, cax = axins, orientation='horizontal')
    # _ax.set_title('$p_{pred}$')

    _ax = fig.add_subplot(_gs[2])
    _ax.axis('off') 
    _h = _ax.imshow(np.abs(p_15[t_gap*2, :, :] - p_01[t_gap*2, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    _h.set_clim(vmin=0, vmax=1)
    axins = inset_axes(_ax,
                width="100%",
                height="10%",
                loc='lower center',
                borderpad=-1,
                )
    fig.colorbar(_h, ax=_ax, ticks=p_error, cax = axins, orientation='horizontal')
    # _ax.set_title('|$p_{ref}$ - $p_{pred}$|')

    plt.savefig('append_fig_lmd_comp.png')
    plt.close()

    l2_error_v = get_relative_l2_error(vel_15, vel_01)
    print(f'mean relative l2 error (vel) full time: {l2_error_v.mean()}')

    l2_error_phi = get_relative_l2_error(phi_15, phi_01)
    print(f'mean relative l2 error (phi) full time: {l2_error_phi.mean()}')

    l2_error_p = get_relative_l2_error(p_15, p_01)
    print(f'mean relative l2 error (p) full time: {l2_error_p.mean()}')

    l2_error_psi1 = get_relative_l2_error(psi_15_1, psi_01_1)
    print(f'mean relative l2 error (psi1) full time: {l2_error_psi1.mean()}')

    l2_error_psi2 = get_relative_l2_error(psi_15_2, psi_01_2)
    print(f'mean relative l2 error (psi2) full time: {l2_error_psi2.mean()}')

    

if __name__ == '__main__':
    main()


















