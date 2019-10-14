#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:10:18 2019

@author: banano
"""

import sys
sys.path.append(u'/Users/banano/Documents/UMD/Research/chern/Analysis2018')
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from importlib import reload
import matplotlib.pyplot as plt
import tqdm
#import utils
#reload(utils)
#from utils import data_processing, make_rois_array, image_filtering, puck, reconstruct_probes
from matplotlib.gridspec import GridSpec
#import Ramsey_analysis_functions as raf
import lmfit
import scipy
from matplotlib import rcParams
from scipy.constants import hbar

#%%
#from scipy.optimize import least_squares

fontsize = 9
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

#%%
Gamma_D2 = 38.11e6 # 1/s
Gamma_D1 = 36.10e6  # 1/s
#I_sat = 

def alpha(wavelength):
    c = 2.99792458 * 1e8
    epsilon_0 = 8.854187817e-12
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    omega_2  = 2 * np.pi * c / (780.24*1e-9)
    omega_1 = 2 * np.pi * c / (794.98*1e-9)
    alpha = Gamma_D2 * 2 / omega_2**3 * (1 / (omega - omega_2) + 0. / (omega + omega_2))
    alpha +=  Gamma_D1 /omega_1**3 * (1/(omega - omega_1)+ 0./(omega + omega_1))
    alpha *= np.pi * c**2/2 
    
    return alpha

def alpha_v(wavelength):
    
    c = 2.99792458 * 1e8
    omega_2  = 2 * np.pi * c / (780.24*1e-9)
    omega_1 = 2 * np.pi * c / (794.98*1e-9)
    omega_0 = (2 * omega_1 + omega_2) / 3
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    alpha_v = alpha(wavelength) / (omega_0 - omega)
    
    return alpha_v


def scattering(wavelength):
    
    c = 2.99792458 * 1e8
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    omega_2  = 2 * np.pi * c / (780.24*1e-9)
    omega_1 = 2 * np.pi * c / (794.98*1e-9)
    scattering = 2 * Gamma_D2**2 / (omega - omega_2)**2 * omega **3 / omega_2**6 
    scattering +=  Gamma_D1**2 / (omega - omega_1)**2 * omega **3 / omega_2**6 
    scattering *= np.pi * c**2 / (2 * hbar)
    
    return scattering 

def u_s(wavelength):
    c = 2.99792458 * 1e8
    omega_2  = 2 * np.pi * c / (780.24*1e-9)
    omega_1 = 2 * np.pi * c / (794.76569*1e-9)
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    d = 3.584e-29 # C/m
    
    u_s = -d ** 2 / 36 * (2/ (omega_2-omega)+1/(omega_1-omega))/hbar
    
    return u_s

def u_v(wavelength):
    c = 2.99792458 * 1e8
    omega_2  = 2 * np.pi * c / (780.24*1e-9)
    omega_1 = 2 * np.pi * c / (794.76569*1e-9)
    omega = 2 * np.pi * c / (wavelength * 1e-9)
    omega_mean = (omega_2 + 2*omega_1)/3
    A_fs = np.abs(omega_2-omega_1)
    Delta_fs = 3 * A_fs / 2
    u_v = 2 * u_s(wavelength) * Delta_fs / (omega_mean - omega)/hbar
    
    return u_v
    
#%%


I = 2*0.6e-3/(np.pi * (150e-6)**2)
c = 2.99792458 * 1e8
epsilon_0 = 8.854187817e-12
E_squared = I / (2 * epsilon_0 * c)

wavelengths = np.linspace(778, 796, int(1e3))
alphas = alpha(wavelengths)
alphas_v = alpha_v(wavelengths) /hbar
alphas_v = u_v(wavelengths) * E_squared


lambdas = np.array([791.09, 792.04, 789, 786, 782, 781 ]) 
omegas =  2 * np.pi / lambdas 
Omegas_norm = np.array([10.54, 14.68, 7.5, 7.94, 25.11, 53.82 ])
Omegas_raw =[6.28, 9.2, 4.4, 5.19, 17.3, 36.25]
tau = np.array([8.99e5, 7.45e5, 1.67e6, 1.63e6, 5498, 808])*1e-6


fig = plt.figure(figsize=(5.9,4.))
#    
gs = GridSpec(2, 2)   
ax = plt.subplot(gs[0,0])

plt.semilogy(wavelengths, np.abs(alphas_v)/3.678e3)
plt.semilogy(lambdas, Omegas_norm/3.278, 'o', mec='k', mfc='lightgray')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Raman coupling strength $[E_{\mathrm{L}}]$')


alpha_min = np.abs(alphas_v[np.argmin(np.abs(alphas))])

plt.xlim([wavelengths.min(), wavelengths.max()])

wavelengths = np.linspace(784, 792, int(1e3))
scatter_rates = scattering(wavelengths) * I
ax = plt.subplot(gs[1,1])

plt.plot(wavelengths, (scatter_rates))
min_idx = np.argmin(scatter_rates)
plt.plot(wavelengths[min_idx], scatter_rates[min_idx], 'o',
         mec='k', mfc='lightgray')
plt.hlines(scatter_rates[min_idx], wavelengths.min(), wavelengths.max(), 
           linestyle='--')
#plt.semilogy(lambdas, 1/tau, 'o', mec='k', ms=7, mfc='lightgray')

plt.xlabel('Wavelength [nm]')
plt.ylabel('Scattering rate [1/s]')
plt.xlim([wavelengths.min(), wavelengths.max()])
#plt.ylim([-100, 10**4])
#plt.savefig('Raman_vs_lambda.pdf')

#     
ax = plt.subplot(gs[1,0])
alphas = alpha(wavelengths)/hbar 
alphas = u_s(wavelengths)/hbar * E_squared 
plt.plot(wavelengths, (alphas))
#y_lim = 0.1
plt.hlines(0,wavelengths.min(), wavelengths.max(), linestyle='--' )
#plt.ylim([-y_lim, y_lim])
min_idx = np.argmin(np.abs(alphas))
plt.plot(wavelengths[min_idx], alphas[min_idx], 'o', mec='k', mfc='lightgray')


plt.xlim([wavelengths.min(), wavelengths.max()])
plt.xlabel('Wavelength [nm]')
plt.ylabel('$\Delta E/\hbar$ [Hz]')

ax = plt.subplot(gs[0,1])
#plt.plot(wavelengths, (alphas))
plt.text(0.2,0.5,'exponential decay data')

plt.tight_layout()
plt.savefig('electric_polarizability.pdf')