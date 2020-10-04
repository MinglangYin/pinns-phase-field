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
   t1 = np.linspace(0.02, 1.16, 20)

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

   v_film_err = [0.00340928, 0.00533428, 0.0092642,  0.01095999, 0.01234244, 0.01350054,
0.0153514,  0.01906389, 0.0282892,  0.02419704, 0.02024339 ,0.02048551,
0.02337741, 0.02656938, 0.03133582, 0.03637136, 0.05411707, 0.05165005,
0.03530834, 0.03809131]
   phi_film_err = [0.00542974, 0.00932204, 0.00961718, 0.00926386, 0.01035573, 0.01011795,
0.0094184,  0.00865329, 0.00828213, 0.00847073, 0.00900357, 0.00957804,
0.00968735, 0.00983862, 0.00959932, 0.00947225, 0.00979333, 0.01075351,
0.01269962, 0.01465273]
   p_film_err = [1.78971626e-06, 2.59875739e-05, 5.39987992e-05, 1.19976655e-04,
1.10924808e-04, 7.44208751e-05, 7.83625908e-05, 9.42237954e-05,
1.24455537e-04, 2.24849536e-04, 4.04101922e-04, 5.76565637e-04,
5.79323576e-04, 4.32695260e-04, 2.96069778e-04, 2.30655248e-04,
2.40053179e-04, 1.95813674e-04, 2.19531616e-04, 4.35281366e-04]
   psi1_film_err = [0.00024218, 0.00036919, 0.00059628, 0.00090825, 0.0012585,  0.00161559,
0.00197037, 0.00231943, 0.00265653, 0.00296641, 0.0032281,  0.00342291,
0.00353938, 0.00359255, 0.00363009, 0.0037109,  0.00387872, 0.00414287,
0.00446733, 0.00479398]
   psi2_film_err = [6.33938437e-05, 8.43920075e-05, 1.22650788e-04, 1.84657738e-04,
2.76506211e-04, 3.97373321e-04, 5.36682688e-04, 6.75192261e-04,
7.93008873e-04, 8.82166223e-04, 9.49718150e-04, 1.00594894e-03,
1.05570269e-03, 1.09565132e-03, 1.12352832e-03, 1.13606984e-03,
1.13306640e-03, 1.12249975e-03, 1.12113894e-03, 1.14694028e-03]

   ## figure phi
   fig = plt.figure(constrained_layout=False, figsize=(18, 6))
   gs = fig.add_gridspec(2, 1)
   gs0 = gs[0].subgridspec(1, 5)
   gs1 = gs[1].subgridspec(1, 5)
   # gs2 = gs[2].subgridspec(1, 2)
   
   ax1 = fig.add_subplot(gs0[0])
   ax2 = fig.add_subplot(gs0[1])
   ax3 = fig.add_subplot(gs0[2])
   ax4 = fig.add_subplot(gs0[3])
   ax5 = fig.add_subplot(gs0[4])

   ax6 = fig.add_subplot(gs1[0])
   ax7 = fig.add_subplot(gs1[1])
   ax8 = fig.add_subplot(gs1[2])
   ax9 = fig.add_subplot(gs1[3])
   ax10 = fig.add_subplot(gs1[4])
   
   ##(0, 0)
   ax1.plot(t, vel_err, label='$\epsilon_{v}$', color='blue')
   ax1.set_ylabel('Error', fontsize=24)
   ax1.set_xlabel('Time (s)', fontsize=24)
   ax1.legend(fontsize=24)

   ax2.plot(t, phi_err, label='$\epsilon_{\phi}$', color='blue')
   ax2.set_xlabel('Time (s)', fontsize=24)
   ax2.legend(fontsize=24)

   ax3.plot(t, p_err, label='$\epsilon_{p}$', color='blue')
   ax3.set_xlabel('Time (s)', fontsize=24)
   ax3.legend(fontsize=24)

   ax4.plot(t, psi1_err, label='$\epsilon_{\psi1}$', color='blue')
   ax4.set_xlabel('Time (s)', fontsize=24)
   ax4.legend(fontsize=24)

   ax5.plot(t, psi2_err, label='$\epsilon_{\psi2}$', color='blue')
   ax5.set_xlabel('Time (s)', fontsize=24)
   ax5.legend(fontsize=24)

   ##(1, 0)
   ax6.plot(t1, v_film_err, label='$\epsilon_{v}$', color='blue')
   ax6.set_ylabel('Error', fontsize=24)
   ax6.set_xlabel('Time (s)', fontsize=24)
   ax6.legend(fontsize=24)

   ax7.plot(t1, phi_film_err, label='$\epsilon_{\phi}$', color='blue')
   ax7.set_xlabel('Time (s)', fontsize=24)
   ax7.legend(fontsize=24)

   ax8.plot(t1, p_film_err, label='$\epsilon_{p}$', color='blue')
   ax8.set_xlabel('Time (s)', fontsize=24)
   ax8.legend(fontsize=24)

   ax9.plot(t1, psi1_film_err, label='$\epsilon_{\psi1}$', color='blue')
   ax9.set_xlabel('Time (s)', fontsize=24)
   ax9.legend(fontsize=24)

   ax10.plot(t1, psi2_film_err, label='$\epsilon_{\psi2}$', color='blue')
   ax10.set_xlabel('Time (s)', fontsize=24)
   ax10.legend(fontsize=24)

   plt.tight_layout()
   plt.savefig('fig_err_time.png')
   plt.close()

if __name__ == '__main__':
    main()

