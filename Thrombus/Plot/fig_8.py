"""
Author: Minglang Yin
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from get_error import *
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
    model_path = '../Inv_K_noise/test_9/CheckPts/model_chkpts.pt'

    file_start = 0
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

    ## figure
    num_x = np.linspace(100, 40000, 101)
    num_data = 5
    ### inferred K
    noise_level = np.linspace(0, 0.2, 5)
    K_inf_text = ['0.006', '0.0049', '0.0060', '0.0054', '0.0051']
    Loss_inf_text = ['0.0058', '0.0061', '0.0109', '0.0161', '0.0277']
    err_phi_text = ['0.00027062720957780217', '0.0009604927461011088', '0.0008397013020615627', '0.0006931294143933145', '0.00112296667331825']
    err_v_text = ['0.009788837434999784', '0.22144974885670984', '0.10303000123009362', '0.18072682246765978', '0.24376745571554478']
    K_inf, Loss_inf, err_phi, err_v = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        K_inf[i] = np.array(K_inf_text[i])
        Loss_inf[i] = np.array(Loss_inf_text[i])
        err_phi[i] = np.array(err_phi_text[i])
        err_v[i] = np.array(err_v_text[i])

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(12,4))
    gs = fig.add_gridspec(1, 1)
    gs0 = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])

    ## ax1: K
    ax1.axhline(y=0.01, linestyle='--', color='black',  label='$K_{ref}$')
    ax1.plot(noise_level, K_inf, color='blue')
    ax1.scatter(noise_level, K_inf, color='blue', label='$K_{inf}$')
    ax1.set_xlabel('noise level', fontsize=18)
    ax1.set_ylabel("$K$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(noise_level)
    ax1.set_yticks(np.linspace(0, 0.02, 5))
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([0, 0.02])
    ax1.set_title('(a)')
    ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(fontsize=14)

    ## ax2
    ax2.plot(noise_level, Loss_inf, color="blue")
    ax2.scatter(noise_level, Loss_inf, color="blue")
    ax2.set_xlabel('noise level', fontsize=18)
    ax2.set_ylabel('Loss', fontsize=18, rotation='horizontal')
    ax2.set_xlim([0, 0.2])
    ax2.set_ylim([0, 0.03])
    ax2.set_xticks(noise_level)
    ax2.set_yticks(np.linspace(0, 0.03, 7))
    ax2.set_aspect('auto')
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(b)')

    ## ax3
    h3, = ax3.plot(noise_level, err_phi, color="blue", label='$\epsilon_{\phi}$')
    ax3.scatter(noise_level, err_phi, color="blue")
    # ax3.plot(noise_level, err_v, color="red", label='$\epsilon_{v}$')
    ax3.set_xlabel('noise level', fontsize=18)
    ax3.set_ylabel('$\epsilon_{\phi}$', fontsize=18, rotation='horizontal')
    ax3.set_title('(c)')
    ax3.set_xlim([0, 0.2])
    ax3.set_ylim([0, 0.003])
    ax3.yaxis.set_label_coords(0.0,1.02)
    
    ax4 = ax3.twinx()
    h4, = ax4.plot(noise_level, err_v, color="red", label='$\epsilon_{v}$')
    ax4.scatter(noise_level, err_v, color="red")
    ax4.set_ylabel('$\epsilon_{v}$',fontsize=18, rotation='horizontal')
    # ax4.set_aspect('auto')
    ax4.set_ylim([0, 0.25])
    ax4.yaxis.set_label_coords(1.0,1.08)
    ax4.legend(handles=[h3, h4],fontsize=14, loc='upper left')
    

    plt.tight_layout()
    fig.savefig('fig_8.png')
    plt.close()


if __name__ == '__main__':
    main()


















