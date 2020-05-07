import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    loss = np.loadtxt('../../Inv_L/test_26/loss_new.dat')
    lmd = np.loadtxt('../../Inv_L/test_26/lmd_new.dat')

    ## figure
    fig = plt.figure(constrained_layout=False, figsize=(8, 4))
    gs = fig.add_gridspec(1, 2)
    ## loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(loss[:, 0], color='blue', linewidth=0.5, label='Loss PDE', alpha=0.7)
    ax1.plot(loss[:, 1]/10, color='black', linewidth=0.5, label='Loss BC', alpha=0.7)
    ax1.plot(loss[:, 2], color='red', linewidth=0.5, label='Loss IC', alpha=0.7)
    ax1.plot(loss[:, 3], color='orange', linewidth=0.5, label='Loss Data', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_ylim([1e-6, 1e2])
    ax1.set_xlim([0, 900000])
    ax1.set_xticks(np.linspace(0, 900000, 4))
    ax1.set_title('(b)')
    ax1.yaxis.set_label_coords(0.0,1.02)
    leg = ax1.legend()
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    # ax1.set_aspect('equal')

    ## lmd
    ax2 = fig.add_subplot(gs[1])
    ax2.axhline(y=0.25, linestyle='--', color='black',  label='true $\lambda_{e}$ = 0.25')
    ax2.plot(lmd, label='inferred $\lambda_{e}$', color='blue')
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\lambda_{e}$', fontsize=18, rotation="horizontal")
    ax2.set_xlim([0, 900000])
    ax2.set_ylim([0, 20])
    ax2.set_xticks(np.linspace(0, 900000, 4))
    ax2.set_title('(c)')
    ax2.legend()
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('fig_mem_his.png')
    plt.close()

if __name__ == '__main__':
    main()