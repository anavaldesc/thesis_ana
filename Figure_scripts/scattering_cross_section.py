#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:37:31 2019

@author: banano
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec     
from matplotlib import rcParams

fontsize = 8
rcParams['axes.labelsize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize

rcParams['pdf.fonttype'] = 42 # True type fonts
#rcParams['font.family'] = 'sans-serif'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}']

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

def sigma(delta, Gamma):
    
    sigma = Gamma / (delta**2+Gamma**2/4)
    
    return sigma

Gamma = 1.0
deltas = np.linspace(-5, 5, 500)

fig = plt.figure(figsize=(2.5, 1.7))
gs = GridSpec(1,1)
ax = plt.subplot(gs[0])
plt.plot(deltas, sigma(deltas, Gamma), color='darkred')
plt.xlabel('$(\omega-\omega_0)/\Gamma$')
plt.ylabel('$\sigma(\omega)$ [arb. u.]')
ax.set_yticklabels([])
ax.set_xlim([-4, 4])
ax.set_xticklabels([-4, -2, 0, 2, 4])
plt.tight_layout()
plt.savefig('scattering_cross_section.pdf')
