"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## figure
    x = [1, 2, 3, 4]
    x1 = [1, 1, 1, 1, 1]
    x2 = [2, 2, 2, 2, 2]
    x3 = [3, 3, 3, 3, 3]
    x4 = [4, 4, 4, 4, 4]
    num_data = 4
    ### inferred K
    num_pt = ['3%', '8%', '13%', '18%']
    case_1_lmd_1 = [3.6877, 2.8672, 0.6234, 2.0931, 2.0931] # 100
    case_1_lmd_2 = [7.8043, 8.5249, 8.3459, 8.3512, 7.8779] # 10000
    case_1_lmd_3 = [7.1987, 7.4398, 7.6937, 8.0195, 7.6072] # 20000
    case_1_lmd_4 = [7.9634, 7.6282, 7.5285, 7.6088, 7.9191] # 30000
    mean_lmd = [np.mean(case_1_lmd_1), np.mean(case_1_lmd_2), np.mean(case_1_lmd_3), np.mean(case_1_lmd_4)]
    print(mean_lmd)

    case_1_err_v_1 = [2.9873855205590667, 3.2127043927584547, 3.3294869820811526, 2.36088935564167, 2.8570469886296075]
    case_1_err_phi_1 = [0.3091649905752977, 0.47575084352113556, 0.5383922781057582, 0.4051489466378461,0.42430191478221635 ]
    case_1_err_p_1 = [0.04794657447929802, 0.08507558885771935, 0.3017403076694168, 0.027124771799574288, 0.029188687999739067]
    case_1_err_psi_1_1 = [0.009214761589751838, 0.0745670444014324, 0.3132362637297167, 0.09369464185517962,0.018701136909236506 ]
    case_1_err_psi_2_1 = [0.016123615542717275, 0.16010166487070124, 0.022668248899909066,0.038792898307636624, 0.0070438898871934275 ]

    case_1_err_v_2 = [0.026382496511332416, 0.02711172229357972,0.029268726182589067 , 0.02373410168157633, 0.0329519599260831 ]
    case_1_err_phi_2 = [0.012767489462616442,0.01318752405326079, 0.014422209734996036,0.01312094044247262, 0.016812794113090068 ]
    case_1_err_p_2 = [7.339196013101132e-05,0.000111968083059762 ,5.747696822145063e-05 , 0.00010380634792653436, 6.267198341108024e-05]
    case_1_err_psi_1_2 = [0.0014247436822070894,0.0023614292672430167 , 0.0008938400932825853,0.0011636234801342842, 0.001919460056542074 ]
    case_1_err_psi_2_2 = [0.0005296830188181887, 0.00048037171100352873, 0.0004664139573527518, 0.00032854611089330163, 0.0005574346078733301]

    case_1_err_v_3 = [0.035862916217276844,0.02786138785061832 ,0.02738049467270402 ,0.025956089599856257,  0.029842818036241024]
    case_1_err_phi_3 =[0.009673767087135608, 0.01111223831536844, 0.010785089564986455,0.011380236476498839, 0.01181394448990461] 
    case_1_err_p_3 = [8.379738623247292e-05, 0.00010723592037299066, 6.497348978757905e-05, 6.0939016003013714e-05, 6.386882097752456e-05]
    case_1_err_psi_1_3 = [0.001230531603090434,0.0019218697925904398 , 0.0026154799880249257, 0.0019133572542728346,0.0022462522206058563 ]
    case_1_err_psi_2_3 = [0.0006135425678543159, 0.000499087973017572, 0.0005132905041227295, 0.0005418013814991564, 0.0005873353130348405]

    case_1_err_v_4 = [0.031796550866461014, 0.024905108937860486,0.02643265143918594 ,0.03004800574028805, 0.023882134449658154 ]
    case_1_err_phi_4 = [0.010717908271449932, 0.010607985404358523, 0.012545894729097626,0.012238134654099854, 0.0120859291598119]
    case_1_err_p_4 = [6.507682992678165e-05, 6.691756914823838e-05, 0.00011825013482405475, 5.636765610314357e-05, 6.509783505012314e-05]
    case_1_err_psi_1_4 = [0.0022420335437448964, 0.0017120684087825143,0.0015176169177993973 , 0.002560373562473837, 0.0022502808137009884]
    case_1_err_psi_2_4 = [0.0008376547837303412,0.000386788388389057 ,0.0007717823732702391 , 0.0006960227363113239, 0.0008003465719040551]

    mean_err_v = [np.mean(case_1_err_v_1), np.mean(case_1_err_v_2), np.mean(case_1_err_v_3), np.mean(case_1_err_v_4)]
    mean_err_phi = [np.mean(case_1_err_phi_1), np.mean(case_1_err_phi_2), np.mean(case_1_err_phi_3), np.mean(case_1_err_phi_4)]
    mean_err_p = [np.mean(case_1_err_p_1), np.mean(case_1_err_p_2), np.mean(case_1_err_p_3), np.mean(case_1_err_p_4)]
    mean_err_psi_1 = [np.mean(case_1_err_psi_1_1), np.mean(case_1_err_psi_1_2), np.mean(case_1_err_psi_1_3), np.mean(case_1_err_psi_1_4)]
    mean_err_psi_2 = [np.mean(case_1_err_psi_2_1), np.mean(case_1_err_psi_2_2), np.mean(case_1_err_psi_2_3), np.mean(case_1_err_psi_2_4)]

    ##### case 2
    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(12,16))
    gs = fig.add_gridspec(4, 1)
    gs0 = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])

    gs1 = gs[1].subgridspec(1, 3)
    ax4 = fig.add_subplot(gs1[0])
    ax5 = fig.add_subplot(gs1[1])
    ax6 = fig.add_subplot(gs1[2])

    ## ax1: lmd
    ax1.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$')
    ax1.scatter(x1, case_1_lmd_1, color='blue')
    ax1.scatter(x2, case_1_lmd_2, color='blue')
    ax1.scatter(x3, case_1_lmd_3, color='blue')
    ax1.scatter(x4, case_1_lmd_4, color='blue')
    ax1.plot(x, mean_lmd, color='black', label='inference mean')

    ax1.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax1.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(np.arange(1, 5))
    ax1.set_xticklabels(num_pt)
    ax1.set_yticks(np.linspace(0, 15, 6))
    # ax1.set_xlim([1,4])
    ax1.set_ylim([0, 15])
    ax1.set_title('(a)')
    # ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend()

    # ## ax2
    ###(full time)
    ax2.scatter(x1, case_1_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax2.scatter(x2, case_1_err_v_2, color='blue', label='$\phi$+u')
    ax2.scatter(x3, case_1_err_v_3, color='blue', label='$\phi$')
    ax2.scatter(x4, case_1_err_v_4, color='blue', label='$\phi$+p')
    ax2.plot(x, mean_err_v, color='black')

    ax2.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax2.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_ylim([1e-3, 1e1])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax3
    ax3.scatter(x1, case_1_err_phi_1, color='blue')
    ax3.scatter(x2, case_1_err_phi_2, color='blue')
    ax3.scatter(x3, case_1_err_phi_3, color='blue')
    ax3.scatter(x4, case_1_err_phi_4, color='blue')
    ax3.plot(x, mean_err_phi, color='black')

    ax3.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax3.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax3.set_xticks(np.arange(1, 5))
    ax3.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax3.set_ylim([1e-3, 1e1])
    ax3.set_yscale('log')
    ax3.set_title('(c)')
    ax3.yaxis.set_label_coords(0.0,1.02)
    # leg3 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    # ## ax4
    ax4.scatter(x1, case_1_err_p_1, color='blue')
    ax4.scatter(x2, case_1_err_p_2, color='blue')
    ax4.scatter(x3, case_1_err_p_3, color='blue')
    ax4.scatter(x4, case_1_err_p_4, color='blue')
    ax4.plot(x, mean_err_p, color='black')

    ax4.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax4.set_ylabel("$\epsilon_{P}$", fontsize=18, rotation='horizontal')
    ax4.set_xticks(np.arange(1, 5))
    ax4.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax4.set_ylim([1e-5, 1e1])
    ax4.set_yscale('log')
    ax4.set_title('(d)')
    ax4.yaxis.set_label_coords(0.0,1.02)
    # leg4 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    # ## ax5
    ax5.scatter(x1, case_1_err_psi_1_1, color='blue')
    ax5.scatter(x2, case_1_err_psi_1_2, color='blue')
    ax5.scatter(x3, case_1_err_psi_1_3, color='blue')
    ax5.scatter(x4, case_1_err_psi_1_4, color='blue')
    ax5.plot(x, mean_err_psi_1, color='black')

    ax5.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax5.set_ylabel("$\epsilon_{\psi1}$", fontsize=18, rotation='horizontal')
    ax5.set_xticks(np.arange(1, 5))
    ax5.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax5.set_ylim([1e-4, 1e1])
    ax5.set_yscale('log')
    ax5.set_title('(e)')
    ax5.yaxis.set_label_coords(0.0,1.02)
    # leg3 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    # ## ax6
    ax6.scatter(x1, case_1_err_psi_2_1, color='blue')
    ax6.scatter(x2, case_1_err_psi_2_2, color='blue')
    ax6.scatter(x3, case_1_err_psi_2_3, color='blue')
    ax6.scatter(x4, case_1_err_psi_2_4, color='blue')
    ax6.plot(x, mean_err_psi_2, color='black')

    ax6.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax6.set_ylabel('$\epsilon_{\psi2}$', fontsize=18, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 5))
    ax6.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax6.set_ylim([1e-5, 1e1])
    ax6.set_yscale('log')
    ax6.set_title('(f)')
    ax6.yaxis.set_label_coords(0.0,1.02)

    
    plt.tight_layout()
    fig.savefig('fig_lmd_num.png')
    plt.close()


if __name__ == '__main__':
    main()


















