"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
    ## parameters
    file_start = 0
    file_end = 20
    dt = 0.03

    num_x1 = 100
    num_x2 = 100
    num_t = file_end-file_start

    x_min, x_max = 0, 6
    y_min, y_max = 0, 2
    asp_ratio = (y_max-y_min)/(x_max-x_min)

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
    ax1.axhline(y=0.01, linestyle='--', color='black',  label='$\kappa_{true}$')
    ax1.plot(noise_level, K_inf, color='blue')
    ax1.scatter(noise_level, K_inf, color='blue', label='$\kappa_{inf}$')
    ax1.set_xlabel('noise level $\sigma$', fontsize=18)
    ax1.set_ylabel("$\kappa$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(noise_level)
    # ax1.set_yticks(np.linspace(1e-4, 1e0, 5))
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([1e-4, 1e0])
    ax1.set_yscale('log')
    ax1.set_title('(a)')
    ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(fontsize=14)

    ## ax2
    ax2.plot(noise_level, Loss_inf, color="blue")
    ax2.scatter(noise_level, Loss_inf, color="blue")
    ax2.set_xlabel('noise level $\sigma$', fontsize=18)
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
    ax3.set_xlabel('noise level $\sigma$', fontsize=18)
    ax3.set_ylabel('$\epsilon_{\phi}$', fontsize=18, rotation='horizontal')
    ax3.set_title('(c)')
    ax3.set_xlim([0, 0.2])
    ax3.set_ylim([0, 0.003])
    ax3.yaxis.set_label_coords(0.0,1.02)
    ax3.tick_params(axis='y', labelcolor='blue')
    
    ax4 = ax3.twinx()
    h4, = ax4.plot(noise_level, err_v, color="red", label='$\epsilon_{v}$')
    ax4.scatter(noise_level, err_v, color="red")
    ax4.set_ylabel('$\epsilon_{v}$',fontsize=18, rotation='horizontal')
    # ax4.set_aspect('auto')
    ax4.set_ylim([0, 0.4])
    ax4.yaxis.set_label_coords(1.0,1.08)
    ax4.legend(handles=[h3, h4],fontsize=18, loc='upper left')
    ax4.tick_params(axis='y', labelcolor='red')
    

    plt.tight_layout()
    fig.savefig('fig_K_noise.png')
    plt.close()


if __name__ == '__main__':
    main()


















