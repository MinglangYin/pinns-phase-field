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
    ## parameters
    ## figure
    x = np.linspace(0.001, 100, 101)
    num_data = 4
    K_inf_text = ['0.0016', '0.0060', '1.9608', '61.8760']
    K_ref_text = ['0.0010', '0.0100', '1.0000', '100.0000']
    vel_err_text = ['0.00485693201145109', '0.004554790202356324', '0.000475935122926176', '0.0011552370839461197']
    phi_err_text = ['0.00014456453697773834', '0.00019076198789733757', '0.00010478658143181054', '8.599519278584236e-05']
    rows = [r'$K_{ref}$', r'$K_{inf}$', r'$||\epsilon_{\phi}||_{2}^{2}$', r'$||\epsilon_{v}||_{2}^{2}$']
    cols = ['case 1', 'case 2', 'case 3', 'case 4']

    K_inf, K_ref, vel_err, phi_err = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        K_inf[i] = np.array(K_inf_text[i])
        K_ref[i] = np.array(K_ref_text[i])
        vel_err[i] = np.array(vel_err_text[i])
        phi_err[i] = np.array(phi_err_text[i])

    K=['0.001', '0.01', '1', '100']

    ## load data
    # loss_k100 = np.loadtxt('../Inv_K/test_7/loss.his')
    K100 = np.loadtxt('../../Inv_K/test_7/K_new.his')

    # loss_k1 = np.loadtxt('../Inv_K/test_6/loss.his')
    K1 = np.loadtxt('../../Inv_K/test_6/K_new.his')

    # loss_k001 = np.loadtxt('../Inv_K/test_5/loss.his')
    K001 = np.loadtxt('../../Inv_K/test_5/K_new.his')

    # loss_k0001 = np.loadtxt('../Inv_K/test_8/loss.his')
    K0001 = np.loadtxt('../../Inv_K/test_8/K_new.his')
    
    ## plotting
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    gs = fig.add_gridspec(2, 4)

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ## plot [0, 0]]
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(x, x, linestyle="--", color="black")
    ax1.plot(K_ref, K_inf, color='blue')
    ax1.scatter(K_ref, K_inf, marker="^", color='blue', s=80, label='$\kappa_{inf}$')

    ax1.set_xlabel('$\kappa_{true}$', fontsize=18)
    ax1.set_ylabel('$\kappa_{inf}$', fontsize=18, rotation="horizontal")
    ax1.set_xticks(K_ref)
    ax1.set_yticks(K_ref)
    ax1.set_title('(a)')
    # ax1.set_xlim([1e-3, 1e2])
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(loc="best",fontsize=14)

    ## ax2
    x = [1, 2, 3, 4]
    h2, = ax2.plot(x, phi_err, color="blue", label='$\epsilon_{\phi}$')
    ax2.scatter(x, phi_err, color="blue")
    # ax3.plot(noise_level, err_v, color="red", label='$\epsilon_{v}$')
    ax2.set_xlabel('$\kappa$', fontsize=18)
    ax2.set_ylabel('$\epsilon_{\phi}$', fontsize=18, rotation='horizontal')
    ax2.set_title('(b)')
    # ax2.set_xlim([1, 4])
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xticklabels(K)
    ax2.set_ylim([0, 0.0010])
    ax2.yaxis.set_label_coords(0.0,1.02)
    
    ax3 = ax2.twinx()
    h3, = ax3.plot(x, vel_err, color="red", label='$\epsilon_{v}$')
    ax3.scatter(x, vel_err, color="red")
    ax3.set_ylabel('$\epsilon_{v}$',fontsize=18, rotation='horizontal')
    ax3.set_aspect('auto')
    ax3.set_ylim([0, 0.01])
    ax3.yaxis.set_label_coords(1.0,1.15)
    ax3.legend(handles=[h2, h3],fontsize=14)

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(K100, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=100, linestyle="--", color="black", label='$\kappa_{true}$')
    ax2.set_ylim((1e-2,1e3))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=14, loc='lower right')
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 600000, 3))
    ax2.set_xlim([0, 600000])
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(c)')

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(K1, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=1, linestyle="--", color="black", label='$\kappa_{true}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=14, loc='lower right')
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 600000, 3))
    ax2.set_xlim([0, 600000])
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(d)')

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,2])
    ax2.plot(K001, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.01, linestyle="--", color="black", label='$\kappa_{true}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=14, loc='best')
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 600000, 3))
    ax2.set_xlim([0, 600000])
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(e)')

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,3])
    ax2.plot(K0001, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{true}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=14, loc='best')
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 600000, 3))
    ax2.set_xlim([0, 600000])
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.set_title('(f)')

    plt.savefig('fig_K_vary.png')
    plt.close()

if __name__ == '__main__':
    main()


















