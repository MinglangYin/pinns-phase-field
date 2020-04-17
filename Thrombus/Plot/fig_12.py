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
    case_1_lmd_1_text = ['0.3837', '9.0906', '9.0095', '8.4321'] # phi + v + psi
    case_1_lmd_2_text = ['0.8292', '4.4607', '4.3409', '3.8645'] # phi + v
    case_1_lmd_3_text = ['1.8785', '1.4981', '1.5486', '1.4635'] # phi

    case_1_err_v_1_text = ['0.5549503156233418', '0.18924386414332572', '0.19081566586107032', '0.19355898201024802']
    case_1_err_v_1_part = [0.3222385644806364, 0.06736448446187007, 0.0568149045244549, 0.06418375019999703]

    case_1_err_phi_1_text = ['0.26071302000725394', '0.02480305501817903', '0.02431983654675567', '0.023939295391819413']
    case_1_err_phi_1_part = [0.3021003500698175, 0.010947907116636037, 0.012089104168307552, 0.01169651714982296]

    case_1_err_v_2_text = ['1.4480560071272341', '0.18142815720224165', '0.2030068741744011', '0.1926480196866232']
    case_1_err_v_2_part = [1.3460084131168213, 0.051762502570097735, 0.05704092441536024, 0.04940382700503209]

    case_1_err_phi_2_text = ['0.302692426705042', '0.020007003775435232', '0.02518701851379984', '0.019223120856477066']
    case_1_err_phi_2_part = [0.3551046662444296, 0.008479998875280243, 0.012473679300320235, 0.009766199055815746]

    case_1_err_v_3_text = ['2.8038325959031645', '0.7488861268498743', '1.002028754294296', '1.1576151389300382']
    case_1_err_v_3_part = [2.5031765627642626, 0.5868397855825642, 0.9308011295411381, 1.0120938960800696]

    case_1_err_phi_3_text = ['0.3711127921882995', '0.016732466632928877', '0.014986603558612518', '0.01598686202356819']
    case_1_err_phi_3_part = [0.5482126471106449, 0.006498067461125888, 0.005223195298955918, 0.007462472026119096]

    case_2_lmd_1_text = ['5.2863', '11.7115', '10.5803', '10.2298'] # phi + v + psi
    case_2_lmd_2_text = ['5.4885', '5.4431', '4.8335', '5.1053'] # phi + v
    case_2_lmd_3_text = ['0.2300', '2.2079', '1.9436', '2.1127'] # phi

    case_2_err_v_1_text = ['0.7020814251401651', '0.17341872473110828', '0.19208910127678727', '0.1695795455832322']
    case_2_err_v_1_part = [0.619459585287426, 0.06167188396660239, 0.06246584590501346, 0.05712597053493553]

    case_2_err_phi_1_text = ['0.2433358404492812', '0.029346691439225935', '0.026744972056635015', '0.02775568285831681']
    case_2_err_phi_1_part = [0.1691861323765793, 0.017148727061063372, 0.01398563349152485, 0.01530289802574764]

    case_2_err_v_2_text = ['1.074889181662402', '0.2128358687321419', '0.19849020940220563', '0.1932277387874229']
    case_2_err_v_2_part = [0.910246696956707, 0.06620552755568819, 0.060788046226501524, 0.05467634248279944]

    case_2_err_phi_2_text = ['0.354995269433386', '0.023087670078115675', '0.020754442189531618', '0.01707718404833896']
    case_2_err_phi_2_part = [0.3999527802908997, 0.011441240134717591, 0.010814691988208294, 0.008303967928485565]

    case_2_err_v_3_text = ['3.5348986294472953', '1.7310836342004225', '1.8278528978256794', '1.8428162012836062']
    case_2_err_v_3_part = [3.2462865752147523, 1.4362462123988275, 1.433792336016671, 1.515291779449468]

    case_2_err_phi_3_text = ['0.2311939399365477', '0.019538994681113354', '0.01388587315712798', '0.013886964386421114']
    case_2_err_phi_3_part = [0.3091823431165195, 0.009172619005834373, 0.006941732948815509, 0.005947477308568197 ]
    

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


















