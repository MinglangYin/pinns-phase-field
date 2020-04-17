"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## load data
    loss_k100 = np.loadtxt('../Inv_K/test_7/loss.his')
    K100 = np.loadtxt('../Inv_K/test_7/K.his')

    loss_k1 = np.loadtxt('../Inv_K/test_6/loss.his')
    K1 = np.loadtxt('../Inv_K/test_6/K.his')

    loss_k001 = np.loadtxt('../Inv_K/test_5/loss.his')
    K001 = np.loadtxt('../Inv_K/test_5/K.his')

    loss_k0001 = np.loadtxt('../Inv_K/test_8/loss.his')
    K0001 = np.loadtxt('../Inv_K/test_8/K.his')

    ## figure
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    gs = fig.add_gridspec(2, 4)

    ## subplot [0, 0]
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(loss_k100, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xticks(np.linspace(0, 300000, 4))
    ax1.set_xlim([0, 300000])
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(K100, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=100, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 300000, 3))
    ax2.set_xlim([0, 400000])
    ax2.yaxis.set_label_coords(0.0,1.02)

    ## subplot [0, 0]
    ax1 = fig.add_subplot(gs[0,1])
    ax1.plot(loss_k1, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xticks(np.linspace(0, 300000, 4))
    ax1.set_xlim([0, 300000])
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,1])
    ax2.plot(K1, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 300000, 3))
    ax2.set_xlim([0, 400000])
    ax2.yaxis.set_label_coords(0.0,1.02)

    ## subplot [0, 0]
    ax1 = fig.add_subplot(gs[0,2])
    ax1.plot(loss_k001, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xticks(np.linspace(0, 300000, 4))
    ax1.set_xlim([0, 300000])
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,2])
    ax2.plot(K001, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 300000, 3))
    ax2.set_xlim([0, 400000])
    ax2.yaxis.set_label_coords(0.0,1.02)

    ## subplot [0, 0]
    ax1 = fig.add_subplot(gs[0,3])
    ax1.plot(loss_k0001, color='blue')
    ax1.set_yscale('log')
    ax1.set_xlabel('epoches', fontsize=18)
    ax1.set_ylabel('loss', fontsize=18, rotation="horizontal")
    ax1.set_xticks(np.linspace(0, 300000, 4))
    ax1.set_xlim([0, 300000])
    ax1.set_aspect('equal')
    ax1.yaxis.set_label_coords(0.0,1.02)

    ## subplot [1, 0]
    ax2 = fig.add_subplot(gs[1,3])
    ax2.plot(K0001, label='$\kappa_{inf}$', color='blue')
    ax2.set_yscale('log')
    ax2.axhline(y=0.001, linestyle="--", color="black", label='$\kappa_{ref}$')
    ax2.set_ylim((1e-4,1e2))
    ax2.set_xlabel('epoches', fontsize=18)
    ax2.set_ylabel('$\kappa$', fontsize=18, rotation="horizontal")
    ax2.legend(fontsize=18)
    ax2.set_aspect('equal')
    ax2.set_xticks(np.linspace(0, 300000, 3))
    ax2.set_xlim([0, 400000])
    ax2.yaxis.set_label_coords(0.0,1.02)


    plt.savefig('append:fig_1.png')
    plt.close()

if __name__ == '__main__':
    main()


















