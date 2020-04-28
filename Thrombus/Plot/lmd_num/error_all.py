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
    case_1_lmd_1 = [1.42, 9.3280, 8.6424, 9.0431] # phi + v + psi
    case_1_lmd_2 = [4.3467, 4.41247, 4.4087, 4.2447] # phi + v
    case_1_lmd_3 = [4.0384, 1.1751, 1.2215, 1.8188] # phi

    case_1_err_v_1 = [0.5684918864962359, 0.014145652170003797, 0.01572135854802598, 0.02621160684646784]
    case_1_err_phi_1 = [0.2885261673175247, 0.01686707358268274, 0.012817297699005697, 0.013595372302310876]
    case_1_err_p_1 = [0.030491674507753058, 0.00010092432738261509, 6.244089967933223e-05, 8.621635307445654e-05]
    case_1_err_psi_1_1 = [0.005231953978333372, 0.0007150437527457803, 0.0010942246314679298, 0.00241645426397665]
    case_1_err_psi_2_1 = [0.0029923287090149395, 0.0001576801918506319, 0.00026570635510500047, 0.00034971347876388924]

    case_1_err_v_2 = [1.5801427980152334, 0.0236740700590421, 0.022105400710721958, 0.025176377316032457]
    case_1_err_phi_2 = [0.34587901315213554, 0.004554589624034001, 0.005477051268135714, 0.006203971294017096]
    case_1_err_p_2 = [0.8987290141562421, 0.023159964304050724, 0.021843947986250344, 0.022815987741877142]
    case_1_err_psi_1_2 = [0.037205108103756726, 0.0004021867419007439,0.0003664290363187488 , 0.0004926839850413341]
    case_1_err_psi_2_2 = [0.07882132528615507, 0.00013824306176476114, 0.00012009634022403823, 0.00012741594666168892 ]

    case_1_err_v_3 = [1.9229535589375566, 0.9780336619134221, 1.0949689814406454, 1.244295351713217]
    case_1_err_phi_3 = [1.0730786094451834, 0.006573887686652532, 0.0031001224300374257, 0.00347579358907108]
    case_1_err_p_3 = [0.8571897416073231, 0.2888816568217975, 0.32406140921066, 0.4109055007628218]
    case_1_err_psi_1_3 = [0.9596741728126401, 0.0014750764581647777, 0.0013436843860961283, 0.0012376463319576476]
    case_1_err_psi_2_3 = [1.154138026696123, 0.0016010700519875107, 0.0012873729021435768, 0.001200807772034765]

    case_2_lmd_1 = [1.5202, 11.2697, 11.2875, 10.2344] # phi + v + psi
    case_2_lmd_2 = [5.1133, 5.6738, 5.5645, 5.0464] # phi + v
    case_2_lmd_3 = [2.1050, 3.7115, 2.3074, 2.0182] # phi

    case_2_err_v_1 = [0.643786333604611, 0.01680473668910079, 0.022184017881027956, 0.024734249842574125]
    case_2_err_phi_1 = [0.19930865358275557, 0.021388486348065984, 0.020309919939100037, 0.01992045387765511]
    case_2_err_p_1 = [0.027973254743016734, 5.806136364976672e-05, 6.614495400030331e-05, 5.8766128717665385e-05]
    case_2_err_psi_1_1 = [0.0055702571280141265,0.000712107601181317 , 0.001299234601110503, 0.0020472078455562315]
    case_2_err_psi_2_1 = [0.014963750881527136, 0.00019339315838802772, 0.00036029754461517087, 0.0002952920501974432]

    case_2_err_v_2 = [0.8213349832740535, 0.04036194387320901, 0.03825191213458989, 0.04465891469106427]
    case_2_err_phi_2 = [0.23900090734976204, 0.0069567196037729736, 0.007204698891111233, 0.006502027887030004]
    case_2_err_p_2 = [0.9780937403296764, 0.017778610714317376, 0.017547665823525672, 0.0199996146501224]
    case_2_err_psi_1_2 = [0.03571946092955265, 0.00047295939582515586, 0.0006787678960762318, 0.0005956629790097419]
    case_2_err_psi_2_2 = [0.026095117403062085, 0.00018474321711445016, 0.0002368453349289654, 0.00022995820944179198]

    case_2_err_v_3 = [3.3698893913614754, 1.7210025180875461, 1.4142157200347274, 1.4085872411938147]
    case_2_err_phi_3 = [0.15489061148890598, 0.009708518564779925, 0.0034930975671129036, 0.0031831653458032467]
    case_2_err_p_3 = [0.8774819694558428, 0.5458893513988161, 0.4500975350212381, 0.45784741331030626]
    case_2_err_psi_1_3 = [0.011652564700262748, 0.001298809757342366, 0.00115275339769769 , 0.0009611254229950619]
    case_2_err_psi_2_3 = [0.005121178342129216, 0.0017153106283618973, 0.0017211084261206208 , 0.0014606540830064216]

    ## plotting
    fig = plt.figure(constrained_layout=False, figsize=(24,8))
    gs = fig.add_gridspec(2, 1)
    gs0 = gs[0].subgridspec(1, 6)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])
    ax4 = fig.add_subplot(gs0[3])
    ax5 = fig.add_subplot(gs0[4])
    ax6 = fig.add_subplot(gs0[5])

    gs1 = gs[1].subgridspec(1, 6)
    ax7 = fig.add_subplot(gs1[0])
    ax8 = fig.add_subplot(gs1[1])
    ax9 = fig.add_subplot(gs1[2])
    ax10 = fig.add_subplot(gs1[3])
    ax11 = fig.add_subplot(gs1[4])
    ax12 = fig.add_subplot(gs1[5])

    ## ax1: lmd
    ax1.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$')
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
    ax1.set_yticks(np.linspace(0, 20, 6))
    # ax1.set_xlim([1,4])
    ax1.set_ylim([0, 20])
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
    # leg2 = ax2.legend([h_, h__],['full time window','partial time window'], loc='upper right')

    # ## ax3
    ax3.plot(x, case_1_err_phi_1, color='blue')
    ax3.scatter(x, case_1_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax3.plot(x, case_1_err_phi_2, color='red')
    ax3.scatter(x, case_1_err_phi_2, color='red', label='$\phi$+u')
    ax3.plot(x, case_1_err_phi_3, color='black')
    ax3.scatter(x, case_1_err_phi_3, color='black', label='$\phi$')

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
    ax4.plot(x, case_1_err_p_1, color='blue')
    ax4.scatter(x, case_1_err_p_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax4.plot(x, case_1_err_p_2, color='red')
    ax4.scatter(x, case_1_err_p_2, color='red', label='$\phi$+u')
    ax4.plot(x, case_1_err_p_3, color='black')
    ax4.scatter(x, case_1_err_p_3, color='black', label='$\phi$')

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
    ax5.plot(x, case_1_err_psi_1_1, color='blue')
    ax5.scatter(x, case_1_err_psi_1_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax5.plot(x, case_1_err_psi_1_2, color='red')
    ax5.scatter(x, case_1_err_psi_1_2, color='red', label='$\phi$+u')
    ax5.plot(x, case_1_err_psi_1_3, color='black')
    ax5.scatter(x, case_1_err_psi_1_3, color='black', label='$\phi$')

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
    ax6.plot(x, case_1_err_psi_2_1, color='blue')
    ax6.scatter(x, case_1_err_psi_2_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax6.plot(x, case_1_err_psi_2_2, color='red')
    ax6.scatter(x, case_1_err_psi_2_2, color='red', label='$\phi$+u')
    ax6.plot(x, case_1_err_psi_2_3, color='black')
    ax6.scatter(x, case_1_err_psi_2_3, color='black', label='$\phi$')

    ax6.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax6.set_ylabel('$\epsilon_{\psi2}$', fontsize=18, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 5))
    ax6.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax6.set_ylim([1e-5, 1e1])
    ax6.set_yscale('log')
    ax6.set_title('(c)')
    ax6.yaxis.set_label_coords(0.0,1.02)
    # leg3 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    
    ## ax7: lmd
    ax7.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$')
    ax7.plot(x, case_2_lmd_1, color='blue')
    ax7.scatter(x, case_2_lmd_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax7.plot(x, case_2_lmd_2, color='red')
    ax7.scatter(x, case_2_lmd_2, color='red', label='$\phi$+u')
    ax7.plot(x, case_2_lmd_3, color='black')
    ax7.scatter(x, case_2_lmd_3, color='black', label='$\phi$')
    ax7.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax7.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax7.set_xticks(np.arange(1, 5))
    ax7.set_xticklabels(num_pt)
    ax7.set_yticks(np.linspace(0, 20, 6))
    # ax1.set_xlim([0, 0.2])
    ax7.set_ylim([0, 20])
    ax7.set_title('(d)')
    # ax1.set_aspect('auto')
    ax7.yaxis.set_label_coords(0.0,1.02)
    ax7.legend()

    # ## ax8
    ax8.plot(x, case_2_err_v_1, color='blue')
    ax8.scatter(x, case_2_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax8.plot(x, case_2_err_v_2, color='red')
    ax8.scatter(x, case_2_err_v_2, color='red', label='$\phi$+u')
    h_, = ax8.plot(x, case_2_err_v_3, color='black')
    ax8.scatter(x, case_2_err_v_3, color='black', label='$\phi$')

    ax8.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax8.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax8.set_xticks(np.arange(1, 5))
    ax8.set_xticklabels(num_pt)
    ax8.set_ylim([1e-3, 1e1])
    ax8.set_yscale('log')
    ax8.set_title('(e)')
    ax8.yaxis.set_label_coords(0.0,1.02)
    # leg5 = ax5.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax5.legend()

    # ## ax9
    ax9.plot(x, case_2_err_phi_1, color='blue')
    ax9.scatter(x, case_2_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax9.plot(x, case_2_err_phi_2, color='red')
    ax9.scatter(x, case_2_err_phi_2, color='red', label='$\phi$+u')
    h_, = ax9.plot(x, case_2_err_phi_3, color='black')
    ax9.scatter(x, case_2_err_phi_3, color='black', label='$\phi$')

    ax9.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax9.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax9.set_xticks(np.arange(1, 5))
    ax9.set_xticklabels(num_pt)
    ax9.set_ylim([1e-3, 1e1])
    ax9.set_yscale('log')
    ax9.set_title('(f)')
    ax9.yaxis.set_label_coords(0.0,1.02)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()
 
    # ## ax10
    ax10.plot(x, case_2_err_p_1, color='blue')
    ax10.scatter(x, case_2_err_p_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax10.plot(x, case_2_err_p_2, color='red')
    ax10.scatter(x, case_2_err_p_2, color='red', label='$\phi$+u')
    h_, = ax10.plot(x, case_2_err_p_3, color='black')
    ax10.scatter(x, case_2_err_p_3, color='black', label='$\phi$')

    ax10.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax10.set_ylabel("$\epsilon_{P}$", fontsize=18, rotation='horizontal')
    ax10.set_xticks(np.arange(1, 5))
    ax10.set_xticklabels(num_pt)
    ax10.set_ylim([1e-5, 1e1])
    ax10.set_yscale('log')
    ax10.set_title('(f)')
    ax10.yaxis.set_label_coords(0.0,1.02)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    # ## ax11
    ax11.plot(x, case_2_err_psi_1_1, color='blue')
    ax11.scatter(x, case_2_err_psi_1_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax11.plot(x, case_2_err_psi_1_2, color='red')
    ax11.scatter(x, case_2_err_psi_1_2, color='red', label='$\phi$+u')
    h_, = ax11.plot(x, case_2_err_psi_1_3, color='black')
    ax11.scatter(x, case_2_err_psi_1_3, color='black', label='$\phi$')

    ax11.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax11.set_ylabel("$\epsilon_{\psi1}$", fontsize=18, rotation='horizontal')
    ax11.set_xticks(np.arange(1, 5))
    ax11.set_xticklabels(num_pt)
    ax11.set_ylim([1e-4, 1e1])
    ax11.set_yscale('log')
    ax11.set_title('(f)')
    ax11.yaxis.set_label_coords(0.0,1.02)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    # ## ax12
    ax12.plot(x, case_2_err_psi_2_1, color='blue')
    ax12.scatter(x, case_2_err_psi_2_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax12.plot(x, case_2_err_psi_2_2, color='red')
    ax12.scatter(x, case_2_err_psi_2_2, color='red', label='$\phi$+u')
    h_, = ax12.plot(x, case_2_err_psi_2_3, color='black')
    ax12.scatter(x, case_2_err_psi_2_3, color='black', label='$\phi$')

    ax12.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax12.set_ylabel("$\epsilon_{\psi2}$", fontsize=18, rotation='horizontal')
    ax12.set_xticks(np.arange(1, 5))
    ax12.set_xticklabels(num_pt)
    ax12.set_ylim([1e-5, 1e1])
    ax12.set_yscale('log')
    ax12.set_title('(f)')
    ax12.yaxis.set_label_coords(0.0,1.02)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    plt.tight_layout()
    fig.savefig('error_all.png')
    plt.close()


if __name__ == '__main__':
    main()


















