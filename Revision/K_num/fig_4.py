"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    K1 = np.loadtxt('./test_17/K.his')

    ## plotting
    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])

    ## subplot [1, 0]
    ax1.plot(K1, label='$\kappa_{inf}$', color='blue')
    ax1.set_yscale('log')
    ax1.axhline(y=1, linestyle="--", color="black", label='$\kappa_{true}$')
    ax1.set_ylim((1e-4,1e3))
    ax1.set_xlabel('epochs', fontsize=18)
    ax1.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax1.legend(fontsize=14, loc='lower right')
    ax1.set_aspect('equal')
    ax1.set_xticks(np.linspace(0, 300000, 3))
    ax1.set_xticklabels(['0', '150K', '300K'])
    ax1.set_xlim([0, 300000])
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.set_title('(c)')

    plt.savefig('fig_K_vary.png')
    plt.close()

if __name__ == '__main__':
    main()


















