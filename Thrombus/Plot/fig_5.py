"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    loss = np.loadtxt('../Inv_K/test_3/loss.his')
    K = np.loadtxt('../Inv_K/test_3/K.his')

    ## figure
    fig = plt.figure(constrained_layout=True, figsize=(9, 4.5))
    gs = fig.add_gridspec(1, 2)
    ## subplot 1
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(loss, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xticks(np.linspace(0, 400000, 5))
    ax1.set_xlim([0, 400000])
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)

    ## subplot 2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(K, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 400000, 5))
    ax2.set_xlim([0, 400000])
    ax2.yaxis.set_label_coords(0.0,1.02)

    plt.savefig('fig_5.png')
    plt.close()

if __name__ == '__main__':
    main()


















