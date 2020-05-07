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
    lmd_inf = [8.1101, 7.2270, 8.4252, 7.3424, 8.0208]
    Loss_inf = [0.1126, 0.1137, 0.1253, 0.1119, 0.1229]
    err_phi = [0.019132969661695337, 0.022165446091459352, 0.028774690058806056, 0.0330374651396011, 0.04862939423576465]
    err_v = [0.06263739293792053, 0.05087910730989601, 0.057199515437603196, 0.05933298259246256, 0.060257633343410756]
    err_p = [0.0006173756011650792, 0.0003939275101607151, 0.0004433617821461121,0.00038196944384585646 ,0.00042253077344307934 ]
    err_psi1 = [0.00539565361006589, 0.0013109016008178505 , 0.0025911403681715415, 0.0014683988697763303,0.002081844771282982 ]
    err_psi2 = [0.0005782863963844629, 0.0004720243355262035, 0.00040608197773290894, 0.0005885836333450958,0.0006157180399772479 ]

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(12,4))
    gs = fig.add_gridspec(1, 1)
    gs0 = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])

    ## ax1: K
    ax1.axhline(y=10, linestyle='--', color='black',  label='actual $\lambda_{e}$ = 10')
    ax1.plot(noise_level, lmd_inf, color='blue')
    ax1.scatter(noise_level, lmd_inf, color='blue', label='inferred $\lambda_{e}$')
    ax1.set_xlabel('noise level $\sigma$', fontsize=18)
    ax1.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(noise_level)
    ax1.set_yticks(np.linspace(0, 20, 6))
    ax1.set_xlim([0, 0.2])
    ax1.set_ylim([0, 20])
    ax1.set_title('(a)')
    ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend()

    ## ax2
    ax2.plot(noise_level, Loss_inf, color="blue")
    ax2.scatter(noise_level, Loss_inf, color="blue")
    ax2.set_xlabel('noise level $\sigma$', fontsize=18)
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
    ax3.plot(noise_level, err_p, color="black", label='$\epsilon_{p}$')
    ax3.scatter(noise_level, err_p, color="black")
    ax3.plot(noise_level, err_psi1, color="gray", label='$\epsilon_{\psi1}$')
    ax3.scatter(noise_level, err_psi1, color="gray")
    ax3.plot(noise_level, err_psi2, color="orange", label='$\epsilon_{\psi2}$')
    ax3.scatter(noise_level, err_psi2, color="orange")

    ax3.set_xlabel('noise level $\sigma$', fontsize=18)
    ax3.set_ylabel('$\epsilon$', fontsize=18, rotation='horizontal')
    ax3.set_title('(c)')
    ax3.set_xlim([0, 0.2])
    ax3.set_ylim([1e-4, 1e0])
    ax3.set_yscale('log')
    ax3.yaxis.set_label_coords(0.0,1.02)
    ax3.legend(fontsize=10, loc='best', ncol=3)
    plt.tight_layout()
    fig.savefig('fig_lmd_noise.png')
    plt.close()


if __name__ == '__main__':
    main()


















