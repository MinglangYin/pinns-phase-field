"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from get_error_b import *
from get_error_a import *

def main():
    ## figure
    x = [1, 2, 3, 4]
    y = np.arange(0, 6)
    num_data = 4

    ## loading model/optimizer state
    case_1_error = np.zeros((6, 4, 5))
    for i in range (1, 7):
        file_1 = '../../Inv_L_1_num/test_' + str(4*i+1) +'/CheckPts/model_chkpts.pt'
        file_2 = '../../Inv_L_1_num/test_' + str(4*i+2) +'/CheckPts/model_chkpts.pt'
        file_3 = '../../Inv_L_1_num/test_' + str(4*i+3) +'/CheckPts/model_chkpts.pt'
        file_4 = '../../Inv_L_1_num/test_' + str(4*i+4) +'/CheckPts/model_chkpts.pt'

        model_path = [file_1, file_2, file_3, file_4]
        case_1_error[i-1, :, :] = get_error_lmd(model_path)

    case_2_error = np.zeros((6, 4, 5))
    for i in range (1, 7):
        file_1 = '../../Inv_L_2_num/test_' + str(4*i+1) +'/CheckPts/model_chkpts.pt'
        file_2 = '../../Inv_L_2_num/test_' + str(4*i+2) +'/CheckPts/model_chkpts.pt'
        file_3 = '../../Inv_L_2_num/test_' + str(4*i+3) +'/CheckPts/model_chkpts.pt'
        file_4 = '../../Inv_L_2_num/test_' + str(4*i+4) +'/CheckPts/model_chkpts.pt'

        model_path = [file_1, file_2, file_3, file_4]
        case_2_error[i-1, :, :] = get_error_lmd_k(model_path)


    ## inferred lmd
    num_pt = ['3%', '8%', '13%', '18%']
    case_1_lmd_1 = [4.3467, 4.41247, 4.4087, 4.2447] # phi + v
    case_1_lmd_2 = [4.0384, 1.1751, 1.2215, 1.8188] # phi
    case_1_lmd_3 = [1.7798, 9.2559, 8.9303, 8.4984] # phi + v + p
    case_1_lmd_4 = [8.2931, 9.4498, 8.9303, 8.6150] # phi + p
    case_1_lmd_5 = [20.0489, 50.8264, 39.7520, 44.7701] # u + p
    case_1_lmd_6 = [9.3547 , 8.1101 , 7.7692, 8.7858] # phi+ p(boundary)

    ##### case 2
    case_2_lmd_1 = [5.1133, 5.6738, 5.5645, 5.0464] # phi + v
    case_2_lmd_2 = [2.1050, 3.7115, 2.3074, 2.0182] # phi
    case_2_lmd_3 = [2.6466, 11.9498, 11.3033, 11.5983] # phi + v + p
    case_2_lmd_4 = [7.8476, 10.2501, 10.4950, 9.9803] # phi + p
    case_2_lmd_5 = [25.0854, 20.6784, 42.6082, 38.8254] # u + p
    case_2_lmd_6 = [12.1322, 8.8930, 8.6074, 11.3140] # phi + p(boundary)

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

    gs2 = gs[2].subgridspec(1, 3)
    ax7 = fig.add_subplot(gs2[0])
    ax8 = fig.add_subplot(gs2[1])
    ax9 = fig.add_subplot(gs2[2])

    gs3 = gs[3].subgridspec(1, 3)
    ax10 = fig.add_subplot(gs3[0])
    ax11 = fig.add_subplot(gs3[1])
    ax12 = fig.add_subplot(gs3[2])

    ## ax1: lmd
    ax1.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$ = 10')
    ax1.plot(x, case_1_lmd_2, color='red', alpha=0.2)
    ax1.scatter(x, case_1_lmd_2, color='red', label='$\phi$', alpha=0.2)
    ax1.plot(x, case_1_lmd_3, color='black', alpha=0.2)
    ax1.scatter(x, case_1_lmd_3, color='black', label='$\phi$+u+p', alpha=0.2)
    ax1.plot(x, case_1_lmd_4, color='orange', alpha=0.2)
    ax1.scatter(x, case_1_lmd_4, color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax1.plot(x, case_1_lmd_6, color='blue', alpha=0.2)
    ax1.scatter(x, case_1_lmd_6, color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)
    ax1.plot(x, case_1_lmd_1, color='green')
    ax1.scatter(x, case_1_lmd_1, color='green', label='$\phi$+u')
    ax1.plot(x, case_1_lmd_5, color='purple')
    ax1.scatter(x, case_1_lmd_5, color='purple', label='u+p')

    ax1.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax1.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(np.arange(1, 5))
    ax1.set_xticklabels(num_pt)
    ax1.set_yticks(np.linspace(0, 80, 6))
    ax1.set_xlim([0.5, 4.5])
    ax1.set_ylim([0, 80])
    ax1.set_title('(a)')
    # ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.fill_between(y, 8, 12, facecolor='gray', alpha=0.2)
    ax1.legend(ncol=2)

    # ## ax2
    ###(full time)
    ax2.plot(x, case_1_error[1, :, 0], color='red', alpha=0.2)
    ax2.scatter(x, case_1_error[1, :, 0], color='red', label='$\phi$', alpha=0.2)
    ax2.plot(x, case_1_error[2, :, 0], color='black', alpha=0.2)
    ax2.scatter(x, case_1_error[2, :, 0], color='black', label='$\phi$+u+p', alpha=0.2)
    ax2.plot(x, case_1_error[3, :, 0], color='orange', alpha=0.2)
    ax2.scatter(x, case_1_error[3, :, 0], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax2.plot(x, case_1_error[5, :, 0], color='blue', alpha=0.2)
    ax2.scatter(x, case_1_error[5, :, 0], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)
    ax2.plot(x, case_1_error[0, :, 0], color='green')
    ax2.scatter(x, case_1_error[0, :, 0], color='green', label='$\phi$+u')
    ax2.plot(x, case_1_error[4, :, 0], color='purple')
    ax2.scatter(x, case_1_error[4, :, 0], color='purple', label='u+p')

    ax2.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax2.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_xlim([0.5,4.5])
    ax2.set_ylim([1e-5, 1e1])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    ax2.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # ax2.legend()

    # ## ax3
    ax3.plot(x, case_1_error[0, :, 1], color='green')
    ax3.scatter(x, case_1_error[0, :, 1], color='green', label='$\phi$+u')
    ax3.plot(x, case_1_error[1, :, 1], color='red', alpha=0.2)
    ax3.scatter(x, case_1_error[1, :, 1], color='red', label='$\phi$', alpha=0.2)
    ax3.plot(x, case_1_error[2, :, 1], color='black', alpha=0.2)
    ax3.scatter(x, case_1_error[2, :, 1], color='black', label='$\phi$+u+p', alpha=0.2)
    ax3.plot(x, case_1_error[3, :, 1], color='orange', alpha=0.2)
    ax3.scatter(x, case_1_error[3, :, 1], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax3.plot(x, case_1_error[4, :, 1], color='purple')
    ax3.scatter(x, case_1_error[4, :, 1], color='purple', label='u+p')
    ax3.plot(x, case_1_error[5, :, 1], color='blue', alpha=0.2)
    ax3.scatter(x, case_1_error[5, :, 1], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax3.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax3.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax3.set_xticks(np.arange(1, 5))
    ax3.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax3.set_xlim([0.5,4.5])
    ax3.set_ylim([1e-5, 1e1])
    ax3.set_yscale('log')
    ax3.set_title('(c)')
    ax3.yaxis.set_label_coords(0.0,1.02)
    ax3.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg3 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    # ## ax4
    ax4.plot(x, case_1_error[0, :, 2], color='green')
    ax4.scatter(x, case_1_error[0, :, 2], color='green', label='$\phi$+u')
    ax4.plot(x, case_1_error[1, :, 2], color='red', alpha=0.2)
    ax4.scatter(x, case_1_error[1, :, 2], color='red', label='$\phi$', alpha=0.2)
    ax4.plot(x, case_1_error[2, :, 2], color='black', alpha=0.2)
    ax4.scatter(x, case_1_error[2, :, 2], color='black', label='$\phi$+u+p', alpha=0.2)
    ax4.plot(x, case_1_error[3, :, 2], color='orange', alpha=0.2)
    ax4.scatter(x, case_1_error[3, :, 2], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax4.plot(x, case_1_error[4, :, 2], color='purple')
    ax4.scatter(x, case_1_error[4, :, 2], color='purple', label='u+p')
    ax4.plot(x, case_1_error[5, :, 2], color='blue', alpha=0.2)
    ax4.scatter(x, case_1_error[5, :, 2], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax4.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax4.set_ylabel("$\epsilon_{P}$", fontsize=18, rotation='horizontal')
    ax4.set_xticks(np.arange(1, 5))
    ax4.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax4.set_xlim([0.5,4.5])
    ax4.set_ylim([1e-5, 1e1])
    ax4.set_yscale('log')
    ax4.set_title('(d)')
    ax4.yaxis.set_label_coords(0.0,1.02)
    ax4.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg4 = ax3.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax3.legend()

    # ## ax5
    ax5.plot(x, case_1_error[0, :, 3], color='green')
    ax5.scatter(x, case_1_error[0, :, 3], color='green', label='$\phi$+u')
    ax5.plot(x, case_1_error[1, :, 3], color='red', alpha=0.2)
    ax5.scatter(x, case_1_error[1, :, 3], color='red', label='$\phi$', alpha=0.2)
    ax5.plot(x, case_1_error[2, :, 3], color='black', alpha=0.2)
    ax5.scatter(x, case_1_error[2, :, 3], color='black', label='$\phi$+u+p', alpha=0.2)
    ax5.plot(x, case_1_error[3, :, 3], color='orange', alpha=0.2)
    ax5.scatter(x, case_1_error[3, :, 3], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax5.plot(x, case_1_error[4, :, 3], color='purple')
    ax5.scatter(x, case_1_error[4, :, 3], color='purple', label='u+p')
    ax5.plot(x, case_1_error[5, :, 3], color='blue', alpha=0.2)
    ax5.scatter(x, case_1_error[5, :, 3], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax5.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax5.set_ylabel("$\epsilon_{\psi1}$", fontsize=18, rotation='horizontal')
    ax5.set_xticks(np.arange(1, 5))
    ax5.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax5.set_xlim([0.5,4.5])
    ax5.set_ylim([1e-5, 1e1])
    ax5.set_yscale('log')
    ax5.set_title('(e)')
    ax5.yaxis.set_label_coords(0.0,1.02)
    ax5.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)

    # ## ax6
    ax6.plot(x, case_1_error[0, :, 4], color='green')
    ax6.scatter(x, case_1_error[0, :, 4], color='green', label='$\phi$+u')
    ax6.plot(x, case_1_error[1, :, 4], color='red', alpha=0.2)
    ax6.scatter(x, case_1_error[1, :, 4], color='red', label='$\phi$', alpha=0.2)
    ax6.plot(x, case_1_error[2, :, 4], color='black', alpha=0.2)
    ax6.scatter(x, case_1_error[2, :, 4], color='black', label='$\phi$+u+p', alpha=0.2)
    ax6.plot(x, case_1_error[3, :, 4], color='orange', alpha=0.2)
    ax6.scatter(x, case_1_error[3, :, 4], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax6.plot(x, case_1_error[4, :, 4], color='purple')
    ax6.scatter(x, case_1_error[4, :, 4], color='purple', label='u+p')
    ax6.plot(x, case_1_error[5, :, 4], color='blue', alpha=0.2)
    ax6.scatter(x, case_1_error[5, :, 4], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax6.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax6.set_ylabel('$\epsilon_{\psi2}$', fontsize=18, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 5))
    ax6.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax6.set_xlim([0.5,4.5])
    ax6.set_ylim([1e-5, 1e1])
    ax6.set_yscale('log')
    ax6.set_title('(f)')
    ax6.yaxis.set_label_coords(0.0,1.02)
    ax6.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)


    ## ax7: lmd
    ax7.axhline(y=10, linestyle='--', color='black',  label='true $\lambda_{e}$ = 10')
    ax7.plot(x, case_2_lmd_2, color='red', alpha=0.2)
    ax7.scatter(x, case_2_lmd_2, color='red', label='$\phi$', alpha=0.2)
    ax7.plot(x, case_2_lmd_3, color='black', alpha=0.2)
    ax7.scatter(x, case_2_lmd_3, color='black', label='$\phi$+u+p', alpha=0.2)
    ax7.plot(x, case_2_lmd_4, color='orange', alpha=0.2)
    ax7.scatter(x, case_2_lmd_4, color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax7.plot(x, case_2_lmd_6, color='blue', alpha=0.2)
    ax7.scatter(x, case_2_lmd_6, color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)
    ax7.plot(x, case_2_lmd_1, color='green')
    ax7.scatter(x, case_2_lmd_1, color='green', label='$\phi$+u')
    ax7.plot(x, case_2_lmd_5, color='purple')
    ax7.scatter(x, case_2_lmd_5, color='purple', label='u+p')

    ax7.fill_between(y, 8, 12, facecolor='gray', alpha=0.2)
    ax7.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax7.set_ylabel("$\lambda_{e}$", fontsize=18, rotation='horizontal')
    ax7.set_xticks(np.arange(1, 5))
    ax7.set_xticklabels(num_pt)
    ax7.set_yticks(np.linspace(0, 80, 6))
    ax7.set_xlim([0.5,4.5])
    ax7.set_ylim([0, 80])
    ax7.set_title('(g)')
    # ax1.set_aspect('auto')
    ax7.yaxis.set_label_coords(0.0,1.02)
    ax7.legend(ncol=2)
    # ax7.text(1.5, 15, "zero $\kappa$", fontsize=16)

    # ## ax8
    ax8.plot(x, case_2_error[0, :, 0], color='green')
    ax8.scatter(x, case_2_error[0, :, 0], color='green', label='$\phi$+u')
    ax8.plot(x, case_2_error[1, :, 0], color='red', alpha=0.2)
    ax8.scatter(x, case_2_error[1, :, 0], color='red', label='$\phi', alpha=0.2)
    ax8.plot(x, case_2_error[2, :, 0], color='black', alpha=0.2)
    ax8.scatter(x, case_2_error[2, :, 0], color='black', label='$\phi$+u+p', alpha=0.2)
    ax8.plot(x, case_2_error[3, :, 0], color='orange', alpha=0.2)
    ax8.scatter(x, case_2_error[3, :, 0], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax8.plot(x, case_2_error[4, :, 0], color='purple')
    ax8.scatter(x, case_2_error[4, :, 0], color='purple', label='u+p')
    ax8.plot(x, case_2_error[5, :, 0], color='blue', alpha=0.2)
    ax8.scatter(x, case_2_error[5, :, 0], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax8.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax8.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax8.set_xticks(np.arange(1, 5))
    ax8.set_xticklabels(num_pt)
    ax8.set_xlim([0.5,4.5])
    ax8.set_ylim([1e-5, 1e1])
    ax8.set_yscale('log')
    ax8.set_title('(h)')
    ax8.yaxis.set_label_coords(0.0,1.02)
    ax8.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg5 = ax5.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax5.legend()

    # ## ax9
    ax9.plot(x, case_2_error[0, :, 1], color='green')
    ax9.scatter(x, case_2_error[0, :, 1], color='green', label='$\phi$+u')
    ax9.plot(x, case_2_error[1, :, 1], color='red', alpha=0.2)
    ax9.scatter(x, case_2_error[1, :, 1], color='red', label='$\phi', alpha=0.2)
    ax9.plot(x, case_2_error[2, :, 1], color='black', alpha=0.2)
    ax9.scatter(x, case_2_error[2, :, 1], color='black', label='$\phi$+u+p', alpha=0.2)
    ax9.plot(x, case_2_error[3, :, 1], color='orange', alpha=0.2)
    ax9.scatter(x, case_2_error[3, :, 1], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax9.plot(x, case_2_error[4, :, 1], color='purple')
    ax9.scatter(x, case_2_error[4, :, 1], color='purple', label='u+p')
    ax9.plot(x, case_2_error[5, :, 1], color='blue', alpha=0.2)
    ax9.scatter(x, case_2_error[5, :, 1], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax9.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax9.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax9.set_xticks(np.arange(1, 5))
    ax9.set_xticklabels(num_pt)
    ax9.set_xlim([0.5,4.5])
    ax9.set_ylim([1e-5, 1e1])
    ax9.set_yscale('log')
    ax9.set_title('(i)')
    ax9.yaxis.set_label_coords(0.0,1.02)
    ax9.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()
 
    # ## ax10
    ax10.plot(x, case_2_error[0, :, 2], color='green')
    ax10.scatter(x, case_2_error[0, :, 2], color='green', label='$\phi$+u')
    ax10.plot(x, case_2_error[1, :, 2], color='red', alpha=0.2)
    ax10.scatter(x, case_2_error[1, :, 2], color='red', label='$\phi', alpha=0.2)
    ax10.plot(x, case_2_error[2, :, 2], color='black', alpha=0.2)
    ax10.scatter(x, case_2_error[2, :, 2], color='black', label='$\phi$+u+p', alpha=0.2)
    ax10.plot(x, case_2_error[3, :, 2], color='orange', alpha=0.2)
    ax10.scatter(x, case_2_error[3, :, 2], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax10.plot(x, case_2_error[4, :, 2], color='purple')
    ax10.scatter(x, case_2_error[4, :, 2], color='purple', label='u+p')
    ax10.plot(x, case_2_error[5, :, 2], color='blue', alpha=0.2)
    ax10.scatter(x, case_2_error[5, :, 2], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax10.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax10.set_ylabel("$\epsilon_{P}$", fontsize=18, rotation='horizontal')
    ax10.set_xticks(np.arange(1, 5))
    ax10.set_xticklabels(num_pt)
    ax10.set_xlim([0.5,4.5])
    ax10.set_ylim([1e-5, 1e1])
    ax10.set_yscale('log')
    ax10.set_title('(j)')
    ax10.yaxis.set_label_coords(0.0,1.02)
    ax10.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    # ## ax11
    ax11.plot(x, case_2_error[0, :, 3], color='green')
    ax11.scatter(x, case_2_error[0, :, 3], color='green', label='$\phi$+u')
    ax11.plot(x, case_2_error[1, :, 3], color='red', alpha=0.2)
    ax11.scatter(x, case_2_error[1, :, 3], color='red', label='$\phi', alpha=0.2)
    ax11.plot(x, case_2_error[2, :, 3], color='black', alpha=0.2)
    ax11.scatter(x, case_2_error[2, :, 3], color='black', label='$\phi$+u+p', alpha=0.2)
    ax11.plot(x, case_2_error[3, :, 3], color='orange', alpha=0.2)
    ax11.scatter(x, case_2_error[3, :, 3], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax11.plot(x, case_2_error[4, :, 3], color='purple')
    ax11.scatter(x, case_2_error[4, :, 3], color='purple', label='u+p')
    ax11.plot(x, case_2_error[5, :, 3], color='blue', alpha=0.2)
    ax11.scatter(x, case_2_error[5, :, 3], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax11.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax11.set_ylabel("$\epsilon_{\psi1}$", fontsize=18, rotation='horizontal')
    ax11.set_xticks(np.arange(1, 5))
    ax11.set_xticklabels(num_pt)
    ax11.set_xlim([0.5,4.5])
    ax11.set_ylim([1e-5, 1e1])
    ax11.set_yscale('log')
    ax11.set_title('(k)')
    ax11.yaxis.set_label_coords(0.0,1.02)
    ax11.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    # ## ax12
    ax12.plot(x, case_2_error[0, :, 4], color='green')
    ax12.scatter(x, case_2_error[0, :, 4], color='green', label='$\phi$+u')
    ax12.plot(x, case_2_error[1, :, 4], color='red', alpha=0.2)
    ax12.scatter(x, case_2_error[1, :, 4], color='red', label='$\phi', alpha=0.2)
    ax12.plot(x, case_2_error[2, :, 4], color='black', alpha=0.2)
    ax12.scatter(x, case_2_error[2, :, 4], color='black', label='$\phi$+u+p', alpha=0.2)
    ax12.plot(x, case_2_error[3, :, 4], color='orange', alpha=0.2)
    ax12.scatter(x, case_2_error[3, :, 4], color='orange', label='$\phi$+$p|_{\Omega}$', alpha=0.2)
    ax12.plot(x, case_2_error[4, :, 4], color='purple')
    ax12.scatter(x, case_2_error[4, :, 4], color='purple', label='u+p')
    ax12.plot(x, case_2_error[5, :, 4], color='blue', alpha=0.2)
    ax12.scatter(x, case_2_error[5, :, 4], color='blue', label='$\phi$+$p|_{\partial\Omega}$', alpha=0.2)

    ax12.set_xlabel('$N_{train}$/$N_{total}$', fontsize=18)
    ax12.set_ylabel("$\epsilon_{\psi2}$", fontsize=18, rotation='horizontal')
    ax12.set_xticks(np.arange(1, 5))
    ax12.set_xticklabels(num_pt)
    ax12.set_xlim([0.5,4.5])
    ax12.set_ylim([1e-5, 1e1])
    ax12.set_yscale('log')
    ax12.set_title('(l)')
    ax12.yaxis.set_label_coords(0.0,1.02)
    ax12.fill_between(y, 1e-5, 5e-2, facecolor='gray', alpha=0.2)
    # leg6 = ax6.legend([h_, h__],['full time window','partial time window'], loc='upper right')
    # ax6.legend()

    plt.tight_layout()
    fig.savefig('append_lmd_num.png')
    plt.close()


if __name__ == '__main__':
    main()


















