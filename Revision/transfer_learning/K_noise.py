"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    loss_old_001 = np.loadtxt('1_001/loss_1.his')
    k_old_001 = np.loadtxt('1_001/K_1.his')
    loss_pre_001 = np.loadtxt('1_001/loss.his')
    k_pre_001 = np.loadtxt('1_001/K.his')

    loss_old_0001 = np.loadtxt('1_0001/loss_1.his')
    k_old_0001 = np.loadtxt('1_0001/K_1.his')
    loss_pre_0001 = np.loadtxt('1_0001/loss.his')
    k_pre_0001 = np.loadtxt('1_0001/K.his')

    epochs = np.linspace(0, 300000, 3)

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(8,8))
    gs = fig.add_gridspec(2, 1)
    gs0 = gs[0].subgridspec(1, 2)
    gs1 = gs[1].subgridspec(1, 2)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs1[0])
    ax4 = fig.add_subplot(gs1[1])

    ## ax1: K
    ax1.axhline(y=0.005, linestyle='--', color='black',  label='$\kappa_{true}$=0.005')
    ax1.plot(k_old_001*0.005, color='blue', label='PINNs')
    ax1.plot(k_pre_001*0.005, color='red', label='Pre-trained PINNs')
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel("$\kappa$", fontsize=14, rotation='horizontal')
    ax1.set_xticks(epochs)
    ax1.set_title('(a)')
    # ax1.set_ylim([-0.1, 0.2])
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(fontsize=14, loc='upper right')

    ## ax2: loss
    ax2.plot(loss_old_001*0.1, color='blue', label='PINNs')
    ax2.plot(loss_pre_001*0.1, color='red', label='Pre-trained PINNs')
    ax2.set_xlabel('Epochs', fontsize=14)
    ax2.set_ylabel("Loss", fontsize=14, rotation='horizontal')
    ax2.set_xticks(epochs)
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.legend(fontsize=14)

    ## ax3: K
    ax3.axhline(y=0.0005, linestyle='--', color='black',  label='$\kappa_{true}$=0.0005')
    ax3.plot(k_old_0001*0.0005, color='blue', label='PINNs')
    ax3.plot(k_pre_0001*0.0005, color='red', label='Pre-trained PINNs')
    ax3.set_xlabel('Epochs', fontsize=14)
    ax3.set_ylabel("$\kappa$", fontsize=14, rotation='horizontal')
    ax3.set_xticks(epochs)
    ax3.set_title('(c)')
    # ax1.set_ylim([-0.1, 0.2])
    ax3.yaxis.set_label_coords(0.0,1.02)
    ax3.legend(fontsize=14, loc='upper right')

    ## ax2: loss
    ax4.plot(loss_old_0001*0.1, color='blue', label='PINNs')
    ax4.plot(loss_pre_0001*0.1, color='red', label='Pre-trained PINNs')
    ax4.set_xlabel('Epochs', fontsize=14)
    ax4.set_ylabel("Loss", fontsize=14, rotation='horizontal')
    ax4.set_xticks(epochs)
    ax4.set_yscale('log')
    ax4.set_title('(d)')
    ax4.yaxis.set_label_coords(0.0,1.02)
    ax4.legend(fontsize=14)


    plt.tight_layout()
    fig.savefig('fig_tl.png')
    plt.close()


if __name__ == '__main__':
    main()


















