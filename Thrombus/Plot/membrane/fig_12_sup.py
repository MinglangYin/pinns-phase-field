"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## figure
    x = np.arange(1, 21)

    case_1_1_err_v = [2.85837266e-03, 5.76856954e-01, 1.40328985e+00, 2.66815029e+00,
 3.16741877e+00, 2.40694258e+00, 1.31281120e+00, 4.84744649e-01,
 5.18412019e-01, 9.18599452e-01, 1.34284676e+00, 2.02234327e+00,
 2.95757492e+00, 2.30897921e+00, 1.17826141e+00, 5.90550404e-01,
 5.40212484e-01, 5.80884078e-01, 1.34776543e+00, 2.63161804e+00]
    case_1_4_err_v = [0.00139807,  0.00982317, 0.02592197, 0.01900205, 0.00968876, 0.01229119,
 0.02896608, 0.05640547, 0.10144683, 0.22909468, 0.20786847, 0.11318003,
 0.03816104,0.04512703, 0.13032146, 0.29304436 ,0.47720766, 0.28610392,
 1.02504068 ,0.74286747]

    case_1_1_err_phi =[5.14073438e-04,2.47972002e-01, 2.48286624e-01, 2.75061364e-01,
 2.86337350e-01, 3.55152899e-01, 4.47892520e-01, 6.23357216e-01,
 6.38914117e-01,4.27558497e-01, 3.55255056e-01, 3.20312616e-01,
 2.75009681e-01 ,2.12977107e-01, 1.53163040e-01, 1.06511710e-01,
 1.10392670e-01, 2.12488335e-01, 3.18894272e-01, 4.37797384e-01]
    case_1_4_err_phi =[0.00494438, 0.00924958, 0.00958649, 0.01116431, 0.01177002, 0.01263882,
 0.01261296, 0.00947454, 0.00762103, 0.00859987, 0.01506704, 0.02461537,
 0.03285619, 0.03553516, 0.03165071, 0.022829,   0.01366469, 0.01359469,
 0.03078156, 0.06620602]

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(8,4))
    gs = fig.add_gridspec(1, 1)
    gs0 = gs[0].subgridspec(1, 2)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])

    ax1.axvline(x=10, linestyle='--', color='black')
    ax1.plot(x, case_1_1_err_v, color='blue', label='% of Points: 3%')
    ax1.plot(x, case_1_4_err_v, color='red', label = '% of Points: 18%')

    ax1.set_xlabel('time step', fontsize=18)
    ax1.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    # ax1.set_xticks(np.arange(1, 21))
    ax1.set_xlim([1, 20])
    ax1.set_xticklabels(['1', '5', '10', '15', '20'])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax1.set_ylim([1e-3, 1e2])
    ax1.set_yscale('log')
    ax1.set_title('(a)')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend()
    ax1.text(1.5, 6, 'partial time window', fontsize=10)
    ax1.fill_between(x[:10], 1e-3, 1e2, facecolor='grey', alpha=0.5)

    ax2.axvline(x=10, linestyle='--', color='black')
    ax2.plot(x, case_1_1_err_phi, color='blue', label='% of Points: 3%')
    ax2.plot(x, case_1_4_err_phi, color='red', label = '% of Points: 18%')
    ax2.set_xlabel('time step', fontsize=18)
    ax2.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    # ax1.set_xticks(np.arange(1, 21))
    ax2.set_xlim([1, 20])
    ax2.set_xticklabels(['1', '5', '10', '15', '20'])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_ylim([1e-3, 1e2])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.legend()
    ax2.text(1.5, 6, 'partial time window', fontsize=10)
    ax2.fill_between(x[:10], 1e-3, 1e2, facecolor='grey', alpha=0.5)

    plt.tight_layout()
    fig.savefig('fig_12_sup.png')
    plt.close()


if __name__ == '__main__':
    main()


















