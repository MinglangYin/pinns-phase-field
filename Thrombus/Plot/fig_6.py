"""
Author: Minglang Yin
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

        self.a = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True, dtype=torch.double) )
        self.b = torch.nn.Parameter(torch.tensor([5], requires_grad=True, dtype=torch.double) )
        self.c = torch.nn.Parameter(torch.tensor([2], requires_grad=True, dtype=torch.double) )
        self.d = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True, dtype=torch.double) )

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
    model_path = '../Inv_K/test_10/CheckPts/model_chkpts.pt'

    file_start = 19
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
        name_u = '../../Data/CH/phase_field_xiaoning/K_1_0001/dat/half' + str(file_id) + '.dat'
        name_phi = '../../Data/CH/phase_field_xiaoning/K_1_0001/dat_phase/halfp' + str(file_id) + '.dat'
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
    print(f'a: {model.a}')
    print(f'b: {model.b}')
    print(f'c: {model.c}')
    print(f'd: {model.d}')
"""
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

    ## figure
    x = np.linspace(0.001, 100, 101)
    num_data = 4
    K_inf_text = ['0.0016', '0.0060', '1.9608', '61.8760']
    K_ref_text = ['0.0010', '0.0100', '1.0000', '100.0000']
    vel_err_text = ['0.0052', '0.0050', '0.0009', '0.0016']
    phi_err_text = ['0.0002', '0.0003', '0.0004', '0.0001']
    rows = [r'$K_{ref}$', r'$K_{inf}$', r'$||\epsilon_{\phi}||_{2}^{2}$', r'$||\epsilon_{v}||_{2}^{2}$']
    cols = ['case 1', 'case 2', 'case 3', 'case 4']

    K_inf, K_ref, vel_err, phi_err = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        K_inf[i] = np.array(K_inf_text[i])
        K_ref[i] = np.array(K_ref_text[i])
        vel_err[i] = np.array(vel_err_text[i])
        phi_err[i] = np.array(phi_err_text[i])

    table_data= [K_ref_text, K_inf_text, vel_err_text, phi_err_text]

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(9, 6))
    gs = fig.add_gridspec(2, 1)

    gs0 = gs[0].subgridspec(1, 20)
    gs1 = gs[1].subgridspec(2, 3) 

    ax1 = fig.add_subplot(gs0[0, 0:6])
    ax2 = fig.add_subplot(gs0[0, 8:20])

    ax3 = fig.add_subplot(gs1[0, 0])
    ax4 = fig.add_subplot(gs1[0, 1])
    ax5 = fig.add_subplot(gs1[0, 2])

    ax6 = fig.add_subplot(gs1[1, 0])
    ax7 = fig.add_subplot(gs1[1, 1])
    ax8 = fig.add_subplot(gs1[1, 2])

    ## plot [0, 0]]
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(x, x, linestyle="--", color="black",label='$K_{ref}$')
    ax1.plot(K_ref, K_inf, color='blue', label='$K_{inf}$')
    ax1.scatter(K_ref, K_inf, marker="^", color='blue', s=80)

    ax1.set_xlabel('$K_{ref}$', fontsize=14)
    ax1.set_ylabel('$K_{inf}$', fontsize=14)
    ax1.set_xticks(K_ref)
    ax1.set_yticks(K_ref)
    ax1.set_title('(a)')
    ax1.grid()
    ax1.legend(loc="best")

    ## plot [0, 1]
    table = ax2.table(cellText=table_data, loc='center', rowLabels=rows, colLabels=cols)
    table.set_fontsize(18)
    table.scale(1,2.5)
    ax2.axis('off')
    ax2.set_title('(b)')
    plt.subplots_adjust(wspace = 0.5)
    plt.tight_layout()

    ## ax3
    v_phi = np.linspace(-1, 1, 3)[:,None]
    v_error = np.linspace(0, 0.1, 3)[:,None]

    ax3.axis('off') 
    h = ax3.imshow(phi_ref[-1, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=-1, vmax=1)
    axins = inset_axes(ax3,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax3.set_title('$\phi_{ref}$')
    fig.colorbar(h, ax=ax3, ticks=v_phi, cax = axins, orientation='horizontal')
    # ax.plot(xt[:,1], xt[:,2], 'kx', label = 'Data (%d points)' % (xt.shape[0]), markersize = 4, clip_on = False)

    ## ax4
    ax4.axis('off') 
    h = ax4.imshow(phi_pred[-1, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=-1, vmax=1)
    axins = inset_axes(ax4,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax4.set_title('$\phi_{inf}$')
    fig.colorbar(h, ax=ax4, ticks=v_phi, cax = axins, orientation='horizontal')

    ## ax5
    ax5.axis('off') 
    h = ax5.imshow(np.abs(phi_pred[-1, :, :] - phi_ref[-1, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=0, vmax=0.1)
    axins = inset_axes(ax5,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax5.set_title('$\epsilon_{\phi}$')
    fig.colorbar(h, ax=ax5, ticks=v_error, cax = axins, orientation='horizontal')

    ## ax6
    ax6.axis('off') 
    v_velocity = np.linspace(0, 0.5, 3)[:,None]
    h = ax6.imshow(vel_pred[-1, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=0, vmax=0.5)
    axins = inset_axes(ax6,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax6.set_title('$v_{ref}$')
    fig.colorbar(h, ax=ax6, ticks=v_velocity, cax = axins, orientation='horizontal')

    ## ax7
    ax7.axis('off') 
    h = ax7.imshow(vel_ref[-1, :, :], interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=0, vmax=0.5)
    axins = inset_axes(ax7,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax7.set_title('$v_{inf}$')
    fig.colorbar(h, ax=ax7, ticks=v_velocity, cax=axins, orientation='horizontal')

    ## ax8
    ax8.axis('off') 
    h = ax8.imshow(np.abs(vel_pred[-1, :, :]-vel_ref[-1, :, :]), interpolation='nearest', cmap='coolwarm', origin='lower', aspect=asp_ratio)
    h.set_clim(vmin=0, vmax=0.1)
    axins = inset_axes(ax8,
                width="100%",  # width = 5% of parent_bbox width
                height="10%",  # height : 50%
                loc='lower center',
                borderpad=-1,
                )
    ax8.set_title('$\epsilon_{v}$')
    fig.colorbar(h, ax=ax8, ticks=v_error, cax=axins, orientation='horizontal')

    plt.gcf().text(0.05, 0.45, "(c)", fontsize=14)

    fig.savefig('fig_5.png')
    plt.close()
"""

if __name__ == '__main__':
    main()


















