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
    fig = plt.figure(constrained_layout=True, figsize=(9, 3.5))
    gs = fig.add_gridspec(1, 2)
    ## subplot 1
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(loss, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=14)
    ax1.set_ylabel('loss', fontsize=14)
    ax1.set_aspect('equal')

    ## subplot 2
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(K, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=14)
    ax2.set_ylabel('$\kappa$', fontsize=14)
    ax2.legend(fontsize=14)
    ax2.set_aspect('equal')

    plt.savefig('fig_4.png')
    plt.close()

if __name__ == '__main__':
    main()


















