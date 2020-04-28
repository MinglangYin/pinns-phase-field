"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def main():
    ## figure
    num_x = np.linspace(100, 40000, 101)
    num_data = 5
    ### inferred K
    noise_level = np.linspace(0, 0.2, 5)
    lmd_inf_text = ['3.7891', '3.9018', '4.0679', '3.7717', '3.8299']
    Loss_inf_text = ['0.0372', '0.0471', '0.0607', '0.0956', '0.1378']
    err_phi_text = ['0.008051419432299447', '0.009478734804247843', '0.01132068173843572', '0.020044809031047436', '0.030512773730721393']
    err_v_text = ['0.02478066552721257', '0.029349231302478702', '0.024614214605236176', '0.02605206420946758', '0.02750679025451804']

    lmd_inf, Loss_inf, err_phi, err_v = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        lmd_inf[i] = np.array(lmd_inf_text[i])
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
    ax1.axhline(y=10, linestyle='--', color='black',  label='real $\lambda_{e}$')
    ax1.plot(noise_level, lmd_inf, color='blue')
    ax1.scatter(noise_level, lmd_inf, color='blue', label='inferred $\lambda_{e}$')
    ax1.set_xlabel('noise level', fontsize=18)
    ax1.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(noise_level)
    ax1.set_yticks(np.linspace(0, 12, 7))
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([0, 12])
    ax1.set_title('(a)')
    ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend()

    ## ax2
    ax2.plot(noise_level, Loss_inf, color="blue")
    ax2.scatter(noise_level, Loss_inf, color="blue")
    ax2.set_xlabel('noise level', fontsize=18)
    ax2.set_ylabel('Loss', fontsize=18, rotation='horizontal')
    ax2.set_xlim([0, 0.2])
    ax2.set_ylim([0, 0.50])
    ax2.set_xticks(noise_level)
    ax2.set_yticks(np.linspace(0, 0.50, 3))
    ax2.set_aspect('auto')
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(b)')

    ## ax3
    h3, = ax3.plot(noise_level, err_phi, color="blue", label='$\epsilon_{\phi}$')
    ax3.scatter(noise_level, err_phi, color="blue")
    ax3.plot(noise_level, err_v, color="red", label='$\epsilon_{v}$')
    ax3.scatter(noise_level, err_v, color="red")
    ax3.set_xlabel('noise level', fontsize=18)
    ax3.set_ylabel('$\epsilon$', fontsize=18, rotation='horizontal')
    ax3.set_title('(c)')
    ax3.set_xlim([0, 0.2])
    ax3.set_ylim([0, 0.05])
    ax3.yaxis.set_label_coords(0.0,1.02)
    ax3.legend(fontsize=14, loc='best')
    
    plt.tight_layout()
    fig.savefig('fig_lmd_noise.png')
    plt.close()


if __name__ == '__main__':
    main()


















