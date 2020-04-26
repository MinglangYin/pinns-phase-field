"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## figure
    x = [1, 2, 3, 4]
    num_data = 4
    ### inferred K
    num_pt = ['3%', '8%', '13%', '18%']
    case_1_lmd_1_text = ['1.42', '9.3280', '8.6424', '9.0431'] # phi + v + psi
    case_1_lmd_2_text = ['4.3467', '3.6599', '3.9970', '3.7015'] # phi + v
    case_1_lmd_3_text = ['4.0384', '0', '1.2215', '1.8188'] # phi

    case_1_err_v_1_text = ['0.5684918864962359', '0.014145652170003797', '0.01572135854802598', '0.02621160684646784']
    case_1_err_v_1_part = [0.30614362118031696, 0.010994302218191759, 0.009654136954058293, 0.01848076806494805]

    case_1_err_phi_1_text = ['0.2885261673175247', '0.01686707358268274', '0.012817297699005697', '0.013595372302310876']
    case_1_err_phi_1_part = [0.31349110259859453, 0.015427493993153107, 0.011441383915495722, 0.0105558451210516]

    case_1_err_v_2_text = ['0.9714725228978958', '0.0305997201060751', '0.024908364899946515', '0.029988923336109547']
    case_1_err_v_2_part = [0.6222528242620611, 0.01894733766036736, 0.018016970006621523, 0.01777284366931691]

    case_1_err_phi_2_text = ['0.1851720216766271', '0.005997608008436245', '0.006356640511934336', '0.0076306061898456885']
    case_1_err_phi_2_part = [0.2207886175015743, 0.005416262470375613, 0.00673990203067343, 0.007403038346134134 ]

    case_1_err_v_3_text = ['1.9229535589375566', '0', '1.0949689814406454', '1.244295351713217']
    case_1_err_v_3_part = [2.1557071982258154, 0, 0.9278781033312177, 0.8407055053883177]

    case_1_err_phi_3_text = ['1.0730786094451834', '0', '0.0031001224300374257', '0.00347579358907108']
    case_1_err_phi_3_part = [1.1313760191274074, 0 , 0.003043233641175828, 0.002931128713575367]

    case_2_lmd_1_text = ['1.5202', '11.2697', '11.2875', '10.2344'] # phi + v + psi
    case_2_lmd_2_text = ['5.1133', '5.6738', '5.5645', '5.0464'] # phi + v
    case_2_lmd_3_text = ['2.1050', '3.7115', '2.3074', '2.0182'] # phi

    case_2_err_v_1_text = ['0.643786333604611', '0.01680473668910079', '0.022184017881027956', '0.024734249842574125']
    case_2_err_v_1_part = [0.5439533980642483, 0.010937453805820594,0.013703663071792391 , 0.01746689706233325]

    case_2_err_phi_1_text = ['0.19930865358275557', '0.021388486348065984', '0.020309919939100037', '0.01992045387765511']
    case_2_err_phi_1_part = [0.14620075235946922, 0.019107204598308544, 0.016617348213591114, 0.01764912206094211 ]

    case_2_err_v_2_text = ['0.8213349832740535', '0.04036194387320901', '0.03825191213458989', '0.04465891469106427']
    case_2_err_v_2_part = [0.7929226530570176, 0.035090147153674596, 0.031609148047386335, 0.036757806207701395]

    case_2_err_phi_2_text = ['0.23900090734976204', '0.0069567196037729736', '0.007204698891111233', '0.006502027887030004']
    case_2_err_phi_2_part = [0.2848037837400863, 0.006532008750213904, 0.0073698303871618685, 0.006075268609609996]

    case_2_err_v_3_text = ['3.3698893913614754', '1.7210025180875461', '1.4142157200347274', '1.4085872411938147']
    case_2_err_v_3_part = [3.036613303090346, 1.2292277888050875, 1.23395107095182, 1.188554367444248]

    case_2_err_phi_3_text = ['0.15489061148890598', '0.009708518564779925', '0.0034930975671129036', '0.0031831653458032467']
    case_2_err_phi_3_part = [0.2094094932029093, 0.009017766113579174, 0.0034442816316000097, 0.002975960034913268]
    

    case_1_lmd_1, case_1_lmd_2, case_1_lmd_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))
    case_1_err_v_1, case_1_err_v_2, case_1_err_v_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))
    case_1_err_phi_1, case_1_err_phi_2, case_1_err_phi_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    case_2_lmd_1, case_2_lmd_2, case_2_lmd_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))
    case_2_err_v_1, case_2_err_v_2, case_2_err_v_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))
    case_2_err_phi_1, case_2_err_phi_2, case_2_err_phi_3 = np.zeros((num_data)), np.zeros((num_data)), np.zeros((num_data))

    for i in range(num_data):
        case_1_lmd_1[i] = np.array(case_1_lmd_1_text[i])
        case_1_lmd_2[i] = np.array(case_1_lmd_2_text[i])
        case_1_lmd_3[i] = np.array(case_1_lmd_3_text[i])
        case_1_err_v_1[i] = np.array(case_1_err_v_1_text[i])
        case_1_err_v_2[i] = np.array(case_1_err_v_2_text[i])
        case_1_err_v_3[i] = np.array(case_1_err_v_3_text[i])
        case_1_err_phi_1[i] = np.array(case_1_err_phi_1_text[i])
        case_1_err_phi_2[i] = np.array(case_1_err_phi_2_text[i])
        case_1_err_phi_3[i] = np.array(case_1_err_phi_3_text[i])

        case_2_lmd_1[i] = np.array(case_2_lmd_1_text[i])
        case_2_lmd_2[i] = np.array(case_2_lmd_2_text[i])
        case_2_lmd_3[i] = np.array(case_2_lmd_3_text[i])
        case_2_err_v_1[i] = np.array(case_2_err_v_1_text[i])
        case_2_err_v_2[i] = np.array(case_2_err_v_2_text[i])
        case_2_err_v_3[i] = np.array(case_2_err_v_3_text[i])
        case_2_err_phi_1[i] = np.array(case_2_err_phi_1_text[i])
        case_2_err_phi_2[i] = np.array(case_2_err_phi_2_text[i])
        case_2_err_phi_3[i] = np.array(case_2_err_phi_3_text[i])

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(12,8))
    gs = fig.add_gridspec(2, 1)
    gs0 = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])

    gs1 = gs[1].subgridspec(1, 3)
    ax4 = fig.add_subplot(gs1[0])
    ax5 = fig.add_subplot(gs1[1])
    ax6 = fig.add_subplot(gs1[2])

    ## ax1: lmd
    ax1.axhline(y=10, linestyle='--', color='black',  label='real $\lambda_{e}$')
    ax1.plot(x, case_1_lmd_1, color='blue')
    ax1.scatter(x, case_1_lmd_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax1.plot(x, case_1_lmd_2, color='red')
    ax1.scatter(x, case_1_lmd_2, color='red', label='$\phi$+u')
    ax1.plot(x, case_1_lmd_3, color='black')
    ax1.scatter(x, case_1_lmd_3, color='black', label='$\phi$')
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
    ax2.plot(x, case_1_err_v_1, color='blue')
    ax2.scatter(x, case_1_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax2.plot(x, case_1_err_v_2, color='red')
    ax2.scatter(x, case_1_err_v_2, color='red', label='$\phi$+u')
    h_, =ax2.plot(x, case_1_err_v_3, color='black')
    ax2.scatter(x, case_1_err_v_3, color='black', label='$\phi$')

    ax2.plot(x, case_1_err_v_1_part, linestyle='--', color='blue')
    ax2.scatter(x, case_1_err_v_1_part, linestyle='--', color='blue')
    ax2.plot(x, case_1_err_v_2_part, linestyle='--', color='red')
    ax2.scatter(x, case_1_err_v_2_part, linestyle='--', color='red')
    h__, = ax2.plot(x, case_1_err_v_3_part, linestyle='--', color='black')
    ax2.scatter(x, case_1_err_v_3_part, linestyle='--', color='black')

    ax2.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax2.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_ylim([1e-2, 1e2])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()
    leg2 = ax2.legend([h_, h__],['full time window','partial time window'], loc='upper right')

    # ## ax3
    ax3.plot(x, case_1_err_phi_1, color='blue')
    ax3.scatter(x, case_1_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax3.plot(x, case_1_err_phi_2, color='red')
    ax3.scatter(x, case_1_err_phi_2, color='red', label='$\phi$+u')
    ax3.plot(x, case_1_err_phi_3, color='black')
    ax3.scatter(x, case_1_err_phi_3, color='black', label='$\phi$')

    ax3.plot(x, case_1_err_phi_1_part, linestyle='--', color='blue')
    ax3.scatter(x, case_1_err_phi_1_part, color='blue', linestyle='--', label='$\phi$+u+p+$\psi$')
    ax3.plot(x, case_1_err_phi_2_part, color='red', linestyle='--')
    ax3.scatter(x, case_1_err_phi_2_part, linestyle='--', color='red', label='$\phi$+u')
    ax3.plot(x, case_1_err_phi_3_part, linestyle='--', color='black')
    ax3.scatter(x, case_1_err_phi_3_part, linestyle='--', color='black', label='$\phi$')

    ax3.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax3.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax3.set_xticks(np.arange(1, 5))
    ax3.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax3.set_ylim([1e-3, 1e1])
    ax3.set_yscale('log')
    ax3.set_title('(c)')
    ax3.yaxis.set_label_coords(0.0,1.02)
    leg3 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()
    
    ## ax4: lmd
    ax4.axhline(y=10, linestyle='--', color='black',  label='real $\lambda_{e}$')
    ax4.plot(x, case_2_lmd_1, color='blue')
    ax4.scatter(x, case_2_lmd_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax4.plot(x, case_2_lmd_2, color='red')
    ax4.scatter(x, case_2_lmd_2, color='red', label='$\phi$+u')
    ax4.plot(x, case_2_lmd_3, color='black')
    ax4.scatter(x, case_2_lmd_3, color='black', label='$\phi$')
    ax4.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax4.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax4.set_xticks(np.arange(1, 5))
    ax4.set_xticklabels(num_pt)
    ax4.set_yticks(np.linspace(0, 20, 6))
    # ax1.set_xlim([0, 0.2])
    ax4.set_ylim([0, 20])
    ax4.set_title('(d)')
    # ax1.set_aspect('auto')
    ax4.yaxis.set_label_coords(0.0,1.02)
    ax4.legend()

    # ## ax5
    ax5.plot(x, case_2_err_v_1, color='blue')
    ax5.scatter(x, case_2_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax5.plot(x, case_2_err_v_2, color='red')
    ax5.scatter(x, case_2_err_v_2, color='red', label='$\phi$+u')
    h_, = ax5.plot(x, case_2_err_v_3, color='black')
    ax5.scatter(x, case_2_err_v_3, color='black', label='$\phi$')
    ## partial time window
    ax5.plot(x, case_2_err_v_1_part, linestyle='--', color='blue')
    ax5.scatter(x, case_2_err_v_1_part, linestyle='--', color='blue', label='$\phi$+u+p+$\psi$')
    ax5.plot(x, case_2_err_v_2_part, linestyle='--', color='red')
    ax5.scatter(x, case_2_err_v_2_part, linestyle='--', color='red', label='$\phi$+u')
    h__, = ax5.plot(x, case_2_err_v_3_part, linestyle='--', color='black')
    ax5.scatter(x, case_2_err_v_3_part, linestyle='--', color='black', label='$\phi$')

    ax5.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax5.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax5.set_xticks(np.arange(1, 5))
    ax5.set_xticklabels(num_pt)
    ax5.set_ylim([1e-2, 1e1])
    ax5.set_yscale('log')
    ax5.set_title('(e)')
    ax5.yaxis.set_label_coords(0.0,1.02)
    leg5 = ax5.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax5.legend()

    # ## ax6
    ax6.plot(x, case_2_err_phi_1, color='blue')
    ax6.scatter(x, case_2_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax6.plot(x, case_2_err_phi_2, color='red')
    ax6.scatter(x, case_2_err_phi_2, color='red', label='$\phi$+u')
    h_, = ax6.plot(x, case_2_err_phi_3, color='black')
    ax6.scatter(x, case_2_err_phi_3, color='black', label='$\phi$')
    ## partial time window
    ax6.plot(x, case_2_err_phi_1_part,  linestyle='--', color='blue')
    ax6.scatter(x, case_2_err_phi_1_part, linestyle='--', color='blue', label='$\phi$+u+p+$\psi$')
    ax6.plot(x, case_2_err_phi_2_part, linestyle='--', color='red')
    ax6.scatter(x, case_2_err_phi_2_part, linestyle='--', color='red', label='$\phi$+u')
    h__, = ax6.plot(x, case_2_err_phi_3_part, linestyle='--', color='black')
    ax6.scatter(x, case_2_err_phi_3_part, linestyle='--', color='black', label='$\phi$')

    ax6.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax6.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 5))
    ax6.set_xticklabels(num_pt)
    ax6.set_ylim([1e-3, 1e0])
    ax6.set_yscale('log')
    ax6.set_title('(f)')
    ax6.yaxis.set_label_coords(0.0,1.02)
    leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()
 
    plt.tight_layout()
    fig.savefig('fig_12.png')
    plt.close()


if __name__ == '__main__':
    main()


















