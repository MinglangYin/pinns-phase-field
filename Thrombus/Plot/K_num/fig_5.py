"""
Author: Minglang Yin
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def main():
    ## figure
    x = [1, 2, 3, 4, 5, 6, 7]
    num_data = 4
    ### inferred K
    num_pt = ['5%', '7.5%', '10%', '15%', '25%', '35%', '45%']

    ## K = 0.001
    K0001_inf = [0.0056, 0.0016, 0.0018, 0.0015, 0.0016, 0.0016, 0.0016] 
    K0001_err_v = [0.03692339637250551, 0.005122691087586185, 0.007008171686910214, 0.00452708413850194, 0.005106505064399686, 0.0052239411916606485, 0.005559307109883266]
    K0001_err_phi = [0.013432807731299406, 0.00015518189842251366, 0.00019363300715335963, 0.00014029456982833175, 0.00017971960099100957, 0.00019347685395596873, 0.00012616948045365877]
  
    ## K = 0.01
    K001_inf = [0.0071, 0.0061, 0.0057,0.0060, 0.0059, 0.0060, 0.0059] 
    K001_err_v = [0.019956355343961698, 0.004366403222419709, 0.004150289836349184,0.004847596176752074 , 0.004716459209682021,0.0051841116386620355 ,0.005292324602118571]
    K001_err_phi = [0.005035145097438842, 0.00012776839726437062, 8.770276449979787e-05, 0.00010100666698216132, 0.00012452675092235863, 0.0003400181973663326, 0.00016427329622591014 ]
  
    ## K = 1
    K1_inf = [33.7858, 2.4510, 0.8944, 0.6366, 1.4161, 1.1031, 0.9322] 
    K1_err_v = [0.0024750954867922502, 0.0012930049895994455,0.0016810380891098327, 0.0007224417908966033,0.0010671751471023263 ,0.0009509310797976431 ,0.0009215035862477997 ]
    K1_err_phi = [0.0032184046463185232, 0.00039639639299529617, 0.00030299826644876097, 0.00032921576497482643,0.0003623800637951874, 0.00043788627544685344, 0.00029464572609123435]
  
    ## K = 100
    K100_inf = [0.4905, 57.3998, 64.2853, 43.8522, 53.4171, 61.8759, 68.5341] 
    K100_err_v = [0.001394808975569186, 0.0011525259135554062, 0.0007193996513891196, 0.001969052625906494, 0.0008976820846366921, 0.0015779864859220075, 0.0007530844482711539 ]
    K100_err_phi = [0.002584556557307176, 0.00028004584566359404, 0.00033853025279850774, 0.00019129848969701648, 0.0003438217614571429, 8.569516600497784e-05, 0.0009250736669903226]
  
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

    ## ax1: K
    ax1.axhline(y=0.001, linestyle='--', color='black',  label='real $\kappa$')
    ax1.plot(x, K0001_inf, color='blue')
    ax1.scatter(x, K0001_inf, color='blue', label='$\kappa_{inf}$')
    ax1.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax1.set_ylabel("$\kappa$", fontsize=18, rotation='horizontal')
    ax1.set_xticks(np.arange(1, 8))
    ax1.set_xticklabels(num_pt)

    ax1.set_xlim([1,7])
    ax1.set_ylim([1e-4, 1e-2])
    ax1.set_title('(a)')
    ax1.set_yscale('log')
    # ax1.set_aspect('auto')
    ax1.yaxis.set_label_coords(0.0,1.02)
    ax1.legend(fontsize=14)

    # ## ax2
    ax2.plot(x, K0001_err_v, color='blue')
    ax2.scatter(x, K0001_err_v, color='blue')
    ax2.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax2.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax2.set_xticks(np.arange(1, 8))
    ax2.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax2.set_xlim([1,7])
    ax2.set_ylim([1e-3, 1e-1])
    ax2.set_yscale('log')
    ax2.set_title('(b)')
    ax2.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax3
    ax3.plot(x, K0001_err_phi, color='blue')
    ax3.scatter(x, K0001_err_phi, color='blue')
    ax3.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax3.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax3.set_xticks(np.arange(1, 8))
    ax3.set_xticklabels(num_pt)
    ax3.set_xlim([1,7])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax3.set_ylim([1e-5, 1e-1])
    ax3.set_yscale('log')
    ax3.set_title('(c)')
    ax3.yaxis.set_label_coords(0.0,1.02)
    # ax3.legend()
    
    ## K = 0.01
    ax4.axhline(y=0.01, linestyle='--', color='black',  label='real $\kappa$')
    ax4.plot(x, K001_inf, color='blue')
    ax4.scatter(x, K001_inf, color='blue', label='$\kappa_{inf}$')
    ax4.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax4.set_ylabel("$\kappa$", fontsize=18, rotation='horizontal')
    ax4.set_xticks(np.arange(1, 8))
    ax4.set_xticklabels(num_pt)
    # ax4.set_yticks(np.linspace(0, 0.02, 11))
    ax4.set_xlim([1,7])
    ax4.set_ylim([1e-3, 1e-1])
    ax4.set_title('(d)')
    ax4.set_yscale('log')
    # ax1.set_aspect('auto')
    ax4.yaxis.set_label_coords(0.0,1.02)
    ax4.legend(fontsize=14)

    # ## ax5
    ax5.plot(x, K001_err_v, color='blue')
    ax5.scatter(x, K001_err_v, color='blue')
    ax5.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax5.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax5.set_xticks(np.arange(1, 8))
    ax5.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax5.set_xlim([1,7])
    ax5.set_ylim([1e-3, 1e-1])
    ax5.set_yscale('log')
    ax5.set_title('(e)')
    ax5.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax3
    ax6.plot(x, K001_err_phi, color='blue')
    ax6.scatter(x, K001_err_phi, color='blue')
    ax6.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax6.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax6.set_xticks(np.arange(1, 8))
    ax6.set_xticklabels(num_pt)
    ax6.set_xlim([1,7])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax6.set_ylim([1e-5, 1e-1])
    ax6.set_yscale('log')
    ax6.set_title('(f)')
    ax6.yaxis.set_label_coords(0.0,1.02)
    # ax3.legend()

    ## K = 1
    ##ax7
    ax7.axhline(y=1, linestyle='--', color='black',  label='real $\kappa$')
    ax7.plot(x, K1_inf, color='blue')
    ax7.scatter(x, K1_inf, color='blue', label='$\kappa_{inf}$')
    ax7.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax7.set_ylabel("$\kappa$", fontsize=18, rotation='horizontal')
    ax7.set_xticks(np.arange(1, 8))
    ax7.set_xticklabels(num_pt)
    # ax7.set_yticks(np.linspace(0, 10, 11))
    ax7.set_xlim([1,7])
    ax7.set_ylim([1e-1, 1e2])
    ax7.set_title('(g)')
    ax7.set_yscale('log')
    # ax1.set_aspect('auto')
    ax7.yaxis.set_label_coords(0.0,1.02)
    ax7.legend(fontsize=14)

    # ## ax8
    ax8.plot(x, K1_err_v, color='blue')
    ax8.scatter(x, K1_err_v, color='blue')
    ax8.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax8.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax8.set_xticks(np.arange(1, 8))
    ax8.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax8.set_xlim([1,7])
    ax8.set_ylim([1e-4, 1e-2])
    ax8.set_yscale('log')
    ax8.set_title('(h)')
    ax8.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax9
    ax9.plot(x, K1_err_phi, color='blue')
    ax9.scatter(x, K1_err_phi, color='blue')
    ax9.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax9.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax9.set_xticks(np.arange(1, 8))
    ax9.set_xticklabels(num_pt)
    ax9.set_xlim([1,7])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax9.set_ylim([1e-4, 1e-2])
    ax9.set_yscale('log')
    ax9.set_title('(i)')
    ax9.yaxis.set_label_coords(0.0,1.02)
    # ax3.legend()

    ## K = 100
    ##ax7
    ax10.axhline(y=100, linestyle='--', color='black',  label='real $\kappa$')
    ax10.plot(x, K100_inf, color='blue')
    ax10.scatter(x, K100_inf, color='blue', label='$\kappa_{inf}$')
    ax10.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax10.set_ylabel("$\kappa$", fontsize=18, rotation='horizontal')
    ax10.set_xticks(np.arange(1, 8))
    ax10.set_xticklabels(num_pt)
    # ax7.set_yticks(np.linspace(0, 10, 11))
    ax10.set_xlim([1,7])
    ax10.set_ylim([1e-1, 1e4])
    ax10.set_title('(j)')
    ax10.set_yscale('log')
    # ax1.set_aspect('auto')
    ax10.yaxis.set_label_coords(0.0,1.02)
    ax10.legend(fontsize=14)

    # ## ax8
    ax11.plot(x, K100_err_v, color='blue')
    ax11.scatter(x, K100_err_v, color='blue')
    ax11.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax11.set_ylabel("$\epsilon_{v}$", fontsize=18, rotation='horizontal')
    ax11.set_xticks(np.arange(1, 8))
    ax11.set_xticklabels(num_pt)
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax11.set_xlim([1,7])
    ax11.set_ylim([1e-4, 1e-2])
    ax11.set_yscale('log')
    ax11.set_title('(k)')
    ax11.yaxis.set_label_coords(0.0,1.02)
    # ax2.legend()

    # ## ax9
    ax12.plot(x, K100_err_phi, color='blue')
    ax12.scatter(x, K100_err_phi, color='blue')
    ax12.set_xlabel('$N_{train}/N_{total}$', fontsize=18)
    ax12.set_ylabel("$\epsilon_{\phi}$", fontsize=18, rotation='horizontal')
    ax12.set_xticks(np.arange(1, 8))
    ax12.set_xticklabels(num_pt)
    ax12.set_xlim([1,7])
    # ax2.set_yticks(np.linspace(0, 12, 6))
    ax12.set_ylim([1e-5, 1e-1])
    ax12.set_yscale('log')
    ax12.set_title('(l)')
    ax12.yaxis.set_label_coords(0.0,1.02)
    # ax3.legend()

    plt.tight_layout()
    fig.savefig('fig_5.png')
    plt.close()


if __name__ == '__main__':
    main()


















