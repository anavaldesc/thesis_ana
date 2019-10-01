# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:12:20 2018

@author: swooty
"""

import numpy as np
import mpmath as mpm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rcParams


# Use me to produce pdf files
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 8

rcParams['pdf.fonttype'] = 42 # True type fonts
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
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


hbar = 1.0545718e-34 # reduced Planck constant m^2 kg/s
mRb =1.44467e-25 #mass of rubidium in kg
k_B = 1.38064852e-23 # Boltzmann constant Joules per degree Kelvin

omega_x = 50.0*2.0*np.pi #Hz
omega_y = 50.0*2.0*np.pi #Hz
omega_z = 50.0*2.0*np.pi #Hz

T=80.0e-9 #temperature in Kelvin 
N=1.0e6 #atom number 

omega_bar = (omega_x*omega_y*omega_z)**(1./3.)
ho_E = hbar*omega_bar

ThermE = T*k_B/ho_E #in h.o. units

E_0 = 0.0#0.5*(omega_x/omega_bar+omega_y/omega_bar+omega_z/omega_bar) #in h.o. units

def deltaN(zeta):
    N_0 = zeta/(1.0-zeta)#zeta/(np.exp(E_0/ThermE)-zeta)
    N_ex = (ThermE**3.0)*mpm.polylog(3,zeta)
    return np.abs(N-float(N_ex)-N_0)

def findZ(zguess):
    d = 1e-10
    deltaN0 = deltaN(zguess)
    alpha = 1e-10
    while deltaN0>10.0:
        deriv = (deltaN(zguess+d)-deltaN(zguess-d))/(2.0*d)
        zguess = zguess - alpha*deriv 
        deltaN0 = deltaN(zguess)
        print zguess, deriv, deltaN0
    return zguess, deltaN0

def n_BE(E_ho,T_ho,zeta):
    n_E = zeta/(np.exp(E_ho/T_ho)-zeta)
    g_E = E_ho**2.0
    return n_E*g_E/2.0
    
    
zetaList = np.linspace(0.9999989, 0.999999,num=500)
deltaNlist = [deltaN(zeta) for zeta in zetaList]
N_0_list = [zeta/(1.0-zeta) for zeta in zetaList]
N_ex_list = [float((ThermE**3.0)*mpm.polylog(3,zeta)) for zeta in zetaList]
#zeta, delta = findZ(0.99999)
#print zeta, delta
#
#
#figure = plt.figure()
#pan=figure.add_subplot(111)
#pan.plot(zetaList,deltaNlist, 'k-', linewidth=2)
#pan.plot(zetaList, N_0_list, 'b-')
#pan.plot(zetaList, N_ex_list, 'r-')
#print np.argmin(deltaNlist)
#print deltaNlist[np.argmin(deltaNlist)]
#print zetaList[np.argmin(deltaNlist)]
#print np.log(zetaList[np.argmin(deltaNlist)])*ThermE
#print N_0_list[np.argmin(deltaNlist)]
#print N_ex_list[np.argmin(deltaNlist)]

#make the figure
plt.close(1)
fig = plt.figure(1,figsize=(5.2,3.0))
gs = gridspec.GridSpec(1,1)
gs.update(left=0.15, right=0.85, top=0.85, bottom = 0.15,wspace = 0.5,hspace=1.0)
pan=fig.add_subplot(gs[0])


bin_size = 50.0 # in h. o. units
E_ho_list = bin_size*np.arange(1000.0/bin_size)+bin_size/2

ThermE=250.0*1e-9*k_B/ho_E
zeta = 0.7414
pan.plot(E_ho_list,n_BE(E_ho_list,ThermE,zeta)*bin_size/1e6,'ro-',label='T=255 nK')
print np.sum(n_BE(E_ho_list,ThermE,zeta)*bin_size)
print zeta/(1.0-zeta)
pan.plot(0.0,zeta/(1.0-zeta)/1e6,'r*')

ThermE=180.0*1e-9*k_B/ho_E
zeta = 0.99999797
pan.plot(E_ho_list,n_BE(E_ho_list,ThermE,zeta)*bin_size/1e6,'bo-',label='T=180 nK')
print np.sum(n_BE(E_ho_list,ThermE,zeta)*bin_size)
print zeta/(1.0-zeta)
pan.plot(0.0,zeta/(1.0-zeta)/1e6,'b*')

ThermE=80.0*1e-9*k_B/ho_E
zeta = 0.99999895
pan.plot(E_ho_list,n_BE(E_ho_list,ThermE,zeta)*bin_size/1e6,'go-',label='T=80 nK')
print np.sum(n_BE(E_ho_list,ThermE,zeta)*bin_size)
print zeta/(1.0-zeta)
pan.plot(0.0,zeta/(1.0-zeta)/1e6,'g*')

plt.legend()
pan.set_ylabel(r'$n(\epsilon)G(\epsilon)/N$')
pan.set_xlabel(r'$\epsilon/\hbar\bar{\omega}$')

plt.savefig('condensation.pdf',transparent=True)