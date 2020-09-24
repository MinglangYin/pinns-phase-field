"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    loss = np.loadtxt('../../Inv_L/test_31/loss.his')
    lmd = np.loadtxt('../../Inv_L/test_31/lmd_e.his')
    K = np.loadtxt('../../Inv_L/test_31/K.his')

    ## figure
    fig = plt.figure(constrained_layout=False, figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    ## loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(loss[:, 0], color='blue', linewidth=0.5, label='Loss PDE', alpha=0.7)
    ax1.plot(loss[:, 1], color='orange', linewidth=0.5, label='Loss BC', alpha=0.7)
    ax1.plot(loss[:, 2], color='red', linewidth=0.5, label='Loss IC', alpha=0.7)
    ax1.plot(loss[:, 3], color='black', linewidth=0.5, label='Loss Data', alpha=0.7)

    ax1.set_yscale('log')
    ax1.set_xlabel('epochs', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xlim([0, 500000])
    ax1.set_ylim([1e-3, 1e2])
    ax1.set_xticks(np.linspace(0, 500000, 3))
    ax1.set_title('(a)')
    ax1.yaxis.set_label_coords(0.0,1.02)
    # ax1.set_aspect('equal')
    leg = ax1.legend(ncol=2, fontsize=10)
    for line in leg.get_lines():
        line.set_linewidth(2.0)

    ## lmd
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$ = 10')
    ax2.plot(lmd, label='inferred $\lambda_{e}$', color='blue')
    ax2.set_xlabel('epochs', fontsize=18)
    ax2.set_ylabel('$\lambda_{e}$', fontsize=18, rotation="horizontal")
    ax2.set_xlim([0, 500000])
    ax2.set_ylim([0, 50])
    ax2.set_xticks(np.linspace(0, 500000, 3))
    ax2.set_title('(b)')
    ax2.legend()
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.set_aspect('equal')

    ## K
    ax3 = fig.add_subplot(gs[2])
    ax3.set_yscale('log')
    ax3.axhline(y=0.01, linestyle='--', color='black',  label='true $K$ = 0.01')
    ax3.plot(K, color='blue', label='inferred $K$')
    ax3.set_xlabel('epochs', fontsize=18)
    ax3.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax3.set_xlim([0, 500000])
    ax3.set_xticks(np.linspace(0, 500000, 4))
    ax3.set_title('(c)')
    ax3.legend()
    ax3.yaxis.set_label_coords(0.0,1.02)

    plt.tight_layout()
    plt.savefig('fig_11.png')
    plt.close()

if __name__ == '__main__':
    main()


















