"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    loss_1 = np.loadtxt('./test_2/loss_1.his')
    loss_2 = np.loadtxt('./test_2/loss_2.his')
    loss_3 = np.loadtxt('./test_2/loss_3.his')
    loss = np.loadtxt('./test_2/loss.his')
    loss = np.concatenate((loss_1, loss_2, loss_3, loss), axis=0)

    lmd_1 = np.loadtxt('./test_2/lmd_e_1.his')
    lmd_2 = np.loadtxt('./test_2/lmd_e_2.his')
    lmd_3 = np.loadtxt('./test_2/lmd_e_3.his')
    lmd = np.loadtxt('./test_2/lmd_e.his')
    lmd = np.concatenate((lmd_1, lmd_2, lmd_3, lmd), axis=0)

    K_1 = np.loadtxt('./test_2/K_1.his')
    K_2 = np.loadtxt('./test_2/K_2.his')
    K_3 = np.loadtxt('./test_2/K_3.his')
    K = np.loadtxt('./test_2/K.his')
    K = np.concatenate((K_1, K_2, K_3, K), axis=0)

    print(K.shape)

    ## figure
    fig = plt.figure(constrained_layout=False, figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    ## loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(loss[:, 0], color='blue', linewidth=0.5, label='Loss PDE', alpha=0.7)
    ax1.plot(loss[:, 1], color='black', linewidth=0.5, label='Loss BC', alpha=0.7)
    ax1.plot(loss[:, 2], color='red', linewidth=0.5, label='Loss IC', alpha=0.7)
    ax1.plot(loss[:, 3], color='orange', linewidth=0.5, label='Loss Data', alpha=0.7)

    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xlim([0, 1200000])
    ax1.set_ylim([1e-3, 1e2])
    ax1.set_xticks(np.linspace(0, 1200000, 4))
    ax1.set_xticklabels(['0', '400K', '800K', '1200K'])
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
    ax2.set_xlabel('Epochs', fontsize=18)
    ax2.set_ylabel('$\lambda_{e}$', fontsize=18, rotation="horizontal")
    ax2.set_xlim([0, 1200000])
    ax2.set_ylim([0, 50])
    ax2.set_xticks(np.linspace(0, 1200000, 4))
    ax2.set_xticklabels(['0', '400K', '800K', '1200K'])
    ax2.set_title('(b)')
    ax2.legend()
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.set_aspect('equal')

    ## K
    ax3 = fig.add_subplot(gs[2])
    ax3.set_yscale('log')
    ax3.axhline(y=0.005, linestyle='--', color='black',  label='true $K$ = 0.005')
    ax3.plot(K, color='blue', label='inferred $K$')
    ax3.set_xlabel('Epochs', fontsize=18)
    ax3.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax3.set_xlim([0, 1200000])
    ax3.set_xticks(np.linspace(0, 1200000, 4))
    ax3.set_xticklabels(['0', '400K', '800K', '1200K'])
    ax3.set_title('(c)')
    ax3.legend()
    ax3.yaxis.set_label_coords(0.0,1.02)

    plt.tight_layout()
    plt.savefig('fig_11.png')
    plt.close()

if __name__ == '__main__':
    main()


















