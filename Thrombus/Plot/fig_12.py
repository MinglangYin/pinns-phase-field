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
    num_pt = ['100', '10000', '20000', '30000']
    case_1_lmd_1_text = ['0.9783', '9.4829', '9.4829', '9.4260'] # phi + v + psi
    case_1_lmd_2_text = ['9.8327', '3.2838', '3.1300', '3.1001'] # phi + v
    case_1_lmd_3_text = ['4.4390', '1.0536', '1.2464', '1.3618'] # phi
    case_1_err_v_1_text = ['1.5289', '0.8532', '0.8482', '0.9106']
    case_1_err_phi_1_text = ['0.2626', '0.0305', '0.0302', '0.0292']
    case_1_err_v_2_text = ['41.8045', '0.7762', '0.9373', '0.7825']
    case_1_err_phi_2_text = ['1.9787', '0.0229', '0.0234', '0.0215']
    case_1_err_v_3_text = ['6.5591', '2.7549', '2.7792', '3.6266']
    case_1_err_phi_3_text = ['0.4471', '0.0191', '0.0171', '0.0198']

    case_2_lmd_1_text = ['11.2741', '13.7730', '12.0356', '11.1850'] # phi + v + psi
    case_2_lmd_2_text = ['7.9832', '5.4431', '4.8335', '5.1053'] # phi + v
    case_2_lmd_3_text = ['0.2300', '2.2079', '1.9436', '2.1127'] # phi
    case_2_err_v_1_text = ['1.6603', '0.7070', '0.8500', '0.7060']
    case_2_err_phi_1_text = ['0.2098', '0.0462', '0.0397', '0.0429']
    case_2_err_v_2_text = ['2.8188', '0.6804', '0.6344', '0.6076']
    case_2_err_phi_2_text = ['0.2493', '0.0231', '0.0208', '0.0172']
    case_2_err_v_3_text = ['7.4614', '3.6012', '3.8402', '3.8856']
    case_2_err_phi_3_text = ['0.2304', '0.0196', '0.0139', '0.0139']

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
    ax1.set_xlabel('number of training points', fontsize=14)
    ax1.set_ylabel("$\lambda_{e}$", fontsize=14, rotation='horizontal')
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
    ax2.plot(x, case_1_err_v_1, color='blue')
    ax2.scatter(x, case_1_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax2.plot(x, case_1_err_v_2, color='red')
    ax2.scatter(x, case_1_err_v_2, color='red', label='$\phi$+u')
    ax2.plot(x, case_1_err_v_3, color='black')
    ax2.scatter(x, case_1_err_v_3, color='black', label='$\phi$')
    ax2.set_xlabel('number of training points', fontsize=14)
    ax2.set_ylabel("$\epsilon_{v}$", fontsize=14, rotation='horizontal')
    ax2.set_xticks(np.arange(1, 5))
    ax2.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_ylim([1e-1, 1e2])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax3
    ax3.plot(x, case_1_err_phi_1, color='blue')
    ax3.scatter(x, case_1_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax3.plot(x, case_1_err_phi_2, color='red')
    ax3.scatter(x, case_1_err_phi_2, color='red', label='$\phi$+u')
    ax3.plot(x, case_1_err_phi_3, color='black')
    ax3.scatter(x, case_1_err_phi_3, color='black', label='$\phi$')
    ax3.set_xlabel('number of training points', fontsize=14)
    ax3.set_ylabel("$\epsilon_{\phi}$", fontsize=14, rotation='horizontal')
    ax3.set_xticks(np.arange(1, 5))
    ax3.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax3.set_ylim([1e-2, 1e1])
    ax3.set_yscale('log')
    ax3.set_title('(b)')
    ax3.yaxis.set_label_coords(0.0,1.02)
    # ax3.legend()
    
    ## ax4: lmd
    ax4.axhline(y=10, linestyle='--', color='black',  label='real $\lambda_{e}$')
    ax4.plot(x, case_2_lmd_1, color='blue')
    ax4.scatter(x, case_2_lmd_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax4.plot(x, case_2_lmd_2, color='red')
    ax4.scatter(x, case_2_lmd_2, color='red', label='$\phi$+u')
    ax4.plot(x, case_2_lmd_3, color='black')
    ax4.scatter(x, case_2_lmd_3, color='black', label='$\phi$')
    ax4.set_xlabel('number of training points', fontsize=14)
    ax4.set_ylabel("$\lambda_{e}$", fontsize=14, rotation='horizontal')
    ax4.set_xticks(np.arange(1, 5))
    ax4.set_xticklabels(num_pt)
    ax4.set_yticks(np.linspace(0, 15, 6))
    # ax1.set_xlim([0, 0.2])
    ax4.set_ylim([0, 15])
    ax4.set_title('(d)')
    # ax1.set_aspect('auto')
    ax4.yaxis.set_label_coords(0.0,1.02)
    # ax4.legend()

    # ## ax5
    ax5.plot(x, case_2_err_v_1, color='blue')
    ax5.scatter(x, case_2_err_v_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax5.plot(x, case_2_err_v_2, color='red')
    ax5.scatter(x, case_2_err_v_2, color='red', label='$\phi$+u')
    ax5.plot(x, case_2_err_v_3, color='black')
    ax5.scatter(x, case_2_err_v_3, color='black', label='$\phi$')
    ax5.set_xlabel('number of training points', fontsize=14)
    ax5.set_ylabel("$\epsilon_{v}$", fontsize=14, rotation='horizontal')
    ax5.set_xticks(np.arange(1, 5))
    ax5.set_xticklabels(num_pt)
    ax5.set_ylim([1e-1, 1e1])
    ax5.set_yscale('log')
    ax5.set_title('(e)')
    ax5.yaxis.set_label_coords(0.0,1.02)
    # ax5.legend()

    # ## ax6
    ax6.plot(x, case_2_err_phi_1, color='blue')
    ax6.scatter(x, case_2_err_phi_1, color='blue', label='$\phi$+u+p+$\psi$')
    ax6.plot(x, case_2_err_phi_2, color='red')
    ax6.scatter(x, case_2_err_phi_2, color='red', label='$\phi$+u')
    ax6.plot(x, case_2_err_phi_3, color='black')
    ax6.scatter(x, case_2_err_phi_3, color='black', label='$\phi$')
    ax6.set_xlabel('number of training points', fontsize=14)
    ax6.set_ylabel("$\epsilon_{\phi}$", fontsize=14, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 5))
    ax6.set_xticklabels(num_pt)
    ax6.set_ylim([1e-2, 1e0])
    ax6.set_yscale('log')
    ax6.set_title('(f)')
    ax6.yaxis.set_label_coords(0.0,1.02)
    # ax6.legend()
 
    plt.tight_layout()
    fig.savefig('fig_12.png')
    plt.close()


if __name__ == '__main__':
    main()


















