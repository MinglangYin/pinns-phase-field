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
    
    ## plotting
    fig = plt.figure(constrained_layout=True, figsize=(9, 4.5))
    gs = fig.add_gridspec(1, 1)

    gs0 = gs[0].subgridspec(1, 2)

    ax1 = fig.add_subplot(gs0[0, 0])
    ax2 = fig.add_subplot(gs0[0, 1])

    ## plot [0, 0]]
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(x, x, linestyle="--", color="black",label='$K_{ref}$')
    ax1.plot(K_ref, K_inf, color='blue')
    ax1.scatter(K_ref, K_inf, marker="^", color='blue', s=80, label='$K_{inf}$')

    ax1.set_xlabel('$K_{ref}$', fontsize=18)
    ax1.set_ylabel('$K_{inf}$', fontsize=18, rotation="horizontal")
    ax1.set_xticks(K_ref)
    ax1.set_yticks(K_ref)
    ax1.set_title('(a)')
    # ax1.set_xlim([1e-3, 1e2])
    ax1.grid()
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(loc="best")

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
    ax2.set_ylim([0, 0.0015])
    ax2.yaxis.set_label_coords(0.0,1.02)
    
    ax3 = ax2.twinx()
    h3, = ax3.plot(x, vel_err, color="red", label='$\epsilon_{v}$')
    ax3.scatter(x, vel_err, color="red")
    ax3.set_ylabel('$\epsilon_{v}$',fontsize=18, rotation='horizontal')
    ax3.set_aspect('auto')
    ax3.set_ylim([0, 0.01])
    ax3.yaxis.set_label_coords(1.0,1.08)
    ax3.legend(handles=[h2, h3],fontsize=18)

    fig.savefig('fig_4.png')
    plt.close()


if __name__ == '__main__':
    main()


















