#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:48:25 2019

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

#%%
#from scipy.optimize import least_squares

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
rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}']
#rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
##]  

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

def n(xi, E):
    
    n = xi / (np.exp(E)-xi)
    
    return n

E_vec = np.linspace(0.0001, 2, 100)
xi_vec =  [0.5, 0.75, 1]
lines = ['-.', '--', '-']

fig = plt.figure(figsize=(3.5,2.5))
gs = GridSpec(1, 1)   
ax = plt.subplot(gs[0])

for i,xi in enumerate(xi_vec):
    plt.plot(E_vec, n(xi,E_vec), 'k', linestyle=lines[i], label='$\zeta=$%s'%xi)
plt.legend()
    
plt.ylim([0,4])
ax.set_xticks([0, 1, 2])
plt.xlim([0,2])
ax.set_xlabel('$E/k_B T$')
ax.set_ylabel('$n(E)$')
plt.tight_layout()
plt.savefig('Bose_distribution.pdf')