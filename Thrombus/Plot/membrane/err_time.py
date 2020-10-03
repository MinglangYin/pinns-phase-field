"""
Author: Minglang Yin
"""
import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
sys.path.insert(0,'../')
from Utils import *
from pinns import *

def main():
   t = np.linspace(0.03, 0.6, 20)
   ##error
   vel_err = [0.00121025, 0.00074912, 0.00074008, 0.00119988, 0.00206117, 0.00330161,
0.00492361, 0.00702406, 0.00985297, 0.01387888, 0.01966338, 0.02758236,
0.03763964, 0.0494523,  0.06226984, 0.07500897, 0.08645403, 0.09541914,
0.10097494, 0.10275603]
   phi_err = [2.27327761e-05, 1.44171200e-05, 4.08636778e-05, 1.24370557e-04,
2.69786306e-04, 4.58706446e-04, 6.53237162e-04, 8.07287988e-04,
8.81671600e-04, 8.56635428e-04, 7.40574482e-04, 5.73892691e-04,
4.28736619e-04, 4.00949870e-04, 5.92464925e-04, 1.08679873e-03,
1.92098292e-03, 3.06390126e-03, 4.40964631e-03, 5.79487742e-03]
   p_err = [0.00039285, 0.00043856, 0.00043083, 0.00040135, 0.00046853, 0.00070048,
0.00109449, 0.00160414, 0.00219377, 0.00284522, 0.00355113, 0.00421636,
0.00474952, 0.00499423, 0.00490717, 0.0043887,  0.00367363, 0.00284322,
0.00210012, 0.0018343 ]
   psi1_err = [0.02641034, 0.02634671, 0.0262116 , 0.0259487,  0.02553761, 0.0250196,
0.02449421, 0.02406479, 0.02381833, 0.02381038, 0.02405791, 0.02454541,
0.025233,   0.02606534, 0.0269839,  0.02793494, 0.02887495, 0.02977573,
0.03062615, 0.03143044,]
   psi2_err = [0.02357703, 0.02354184, 0.02343036, 0.02321053, 0.02286329, 0.02238893,
0.02182618, 0.02126197, 0.02082008, 0.02063043, 0.02078928, 0.02133195,
0.02223416, 0.02344263, 0.02490741, 0.02659426, 0.02848326, 0.03056472,
0.03283817, 0.0353174 ]


   ## figure phi
   fig = plt.figure(constrained_layout=False, figsize=(7, 9))
   gs = fig.add_gridspec(3, 1)
   gs0 = gs[0].subgridspec(1, 2)
   gs1 = gs[1].subgridspec(1, 2)
   gs2 = gs[2].subgridspec(1, 2)
   
   ax1 = fig.add_subplot(gs0[0])
   ax2 = fig.add_subplot(gs0[1])
   ax3 = fig.add_subplot(gs1[0])
   ax4 = fig.add_subplot(gs1[1])
   ax5 = fig.add_subplot(gs2[0])
   
   ##(0, 0)
   ax1.plot(t, vel_err, label='$\epsilon_{v}$', color='blue')
   ax1.set_title('(a)')
   ax1.set_ylabel('Error')
   ax1.set_xlabel('Time (s)')
   ax1.legend()

   ax2.plot(t, phi_err, label='$\epsilon_{\phi}$', color='blue')
   ax2.set_title('(b)')
   # ax2.set_ylabel('Error')
   ax2.set_xlabel('Time (s)')
   ax2.legend()

   ax3.plot(t, p_err, label='$\epsilon_{p}$', color='blue')
   ax3.set_title('(c)')
   ax3.set_ylabel('Error')
   ax3.set_xlabel('Time (s)')
   ax3.legend()

   ax4.plot(t, psi1_err, label='$\epsilon_{\psi1}$', color='blue')
   ax4.set_title('(d)')
   # ax4.set_ylabel('Error')
   ax4.set_xlabel('Time (s)')
   ax4.legend()

   ax5.plot(t, psi2_err, label='$\epsilon_{\psi2}$', color='blue')
   ax5.set_title('(e)')
   ax5.set_ylabel('Error')
   ax5.set_xlabel('Time (s)')
   ax5.legend()
   
   plt.tight_layout()
   plt.savefig('fig_err_time.png')
   plt.close()

if __name__ == '__main__':
    main()

