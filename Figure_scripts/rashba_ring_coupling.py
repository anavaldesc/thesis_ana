#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:07:53 2019

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams


fontsize = 9
rcParams['axes.labelsize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize

rcParams['pdf.fonttype'] = 42 # True type fonts
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = [r'\usepackage{cmbright}', r'\usepackage{amsmath}']
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


rcParams['axes.linewidth'] = 0.75
rcParams['lines.linewidth'] = 0.75

rcParams['xtick.major.size'] = 3      # major tick size in points
rcParams['xtick.minor.size'] = 2      # minor tick size in points
rcParams['xtick.major.width'] = 0.75       # major tick width in points
rcParams['xtick.minor.width'] = 0.75      # minor tick width in points

rcParams['ytick.major.size'] = 3      # major tick size in points
rcParams['ytick.minor.size'] = 2      # minor tick size in points
rcParams['ytick.major.width'] = 0.75       # major tick width in points
rcParams['ytick.minor.width'] = 0.75      # minor tick width in points

x = np.linspace(0, 3, 2**8)
y = np.cos(2*np.pi * x / 3)
x_l = np.array([1, 2, 3])
y_l = np.cos(2*np.pi * x_l / 3)

fig = plt.figure(figsize=(5.5/1.5,3.7/1.5))
plt.plot(x, y, linewidth=1.5)
plt.plot(x_l, y_l, 'o', mec='k', mfc='whitesmoke', ms=7)

plt.xticks([0, 1, 2, 3])
plt.yticks([-1, 0, 1], ['$-2\Omega$', '$0$', '$2\Omega$'])
plt.xlabel('$l$')
plt.ylabel('$E_l$')
plt.grid(True)
plt.tight_layout()
plt.savefig('ring_coupling_energies.pdf', transparent=True)
plt.show()
print('banana')