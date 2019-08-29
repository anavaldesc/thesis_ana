#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:12:31 2019

@author: banano
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:08:33 2019

@author: Ana
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eigh
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

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
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
#]  

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

def H_RashbaRF(qx, qy, theta_rot, 
               Omega1, Omega2, Omega3, 
               delta1, delta2, delta3, regular=False):
    
    k1_x = np.cos( 0.5 * np.pi + theta_rot) 
    k1_y = -np.sin(0.5 * np.pi + theta_rot) 
    k2_x = np.cos(1.75 * np.pi + theta_rot) 
    k2_y = -np.sin(1.75 * np.pi + theta_rot) 
    k3_x = np.cos(1.25 * np.pi + theta_rot)
    k3_y = -np.sin(1.25 * np.pi + theta_rot)
    
    if not regular:
    
        k1_x = np.cos( 91.04 * np.pi / 180) 
        k1_y = np.sin(91.04 * np.pi / 180) 
        k2_x = np.cos(317.95 * np.pi / 180) 
        k2_y = np.sin(317.95 * np.pi / 180) 
        k3_x = np.cos(227.64 * np.pi / 180)
        k3_y = np.sin(227.64 * np.pi / 180)
        
    else:
        k1_x = np.cos(0 * np.pi / 180) 
        k1_y = np.sin(0 * np.pi / 180) 
        k2_x = np.cos(120 * np.pi / 180) 
        k2_y = np.sin(120 * np.pi / 180) 
        k3_x = np.cos(240 * np.pi / 180)
        k3_y = np.sin(240 * np.pi / 180)
        
#    
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 , 1.0j*Omega1, Omega3], 
          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2+delta2, -1.0j*Omega2], 
          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2+delta3]])
    H = np.array(H, dtype='complex')

    return H 

#%%
k = np.linspace(-1, 1, 2**6)

def energies(idx, **kwargs):

    return eigvalsh(H_RashbaRF(**kwargs))[idx]

def eigenvec(idx, **kwargs):
    return eigh(H_RashbaRF(**kwargs))[1][:,idx]

eigarray = np.vectorize(energies)
#eigenvecs = np.vectorize(eigenvec)

q = np.linspace(-2, 2, 100)
q2 = np.linspace(-2, 0, 100)
qx, qy = np.meshgrid(q, q2)

start=1
q_r = np.linspace(0, 2, 100)
q_theta = np.linspace(start*np.pi, (1+start)*np.pi, 100)
q_theta = np.linspace(0, 2*np.pi, 100)
radius_matrix, theta_matrix = np.meshgrid(q_r,q_theta)
#qx = radius_matrix * np.cos(theta_matrix)
#qy = radius_matrix * np.sin(theta_matrix)

#qx=np.linspace(-2, 2, 100)
#qy=np.linspace(-2, 2, 100)
x=np.pi*start
theta_rot=3*2*np.pi/360
Omega = 1.75
Omega1=Omega
Omega2=Omega
Omega3=Omega
delta1=0
delta2=0
delta3=0
#

kwargs = {'qx':qx, 'qy': qy, 'theta_rot':theta_rot, 
          'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3}
e0 = eigarray(0, **kwargs)


q_r = np.linspace(0, 0.25, 100)
radius_matrix, theta_matrix = np.meshgrid(q_r,q_theta)
qx2 = radius_matrix * np.cos(theta_matrix)
qy2 = radius_matrix * np.sin(theta_matrix)
e1 = eigarray(1, **kwargs)
e2 = eigarray(2, **kwargs)

kwargs = {'qx':qx, 'qy': qy, 'theta_rot':theta_rot, 
          'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3}


kwargs = {'qx':q*np.cos(x), 'qy': q*np.sin(x), 'theta_rot':theta_rot, 
          'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3}

e0_line = eigarray(0, **kwargs)
e1_line = eigarray(1, **kwargs)
e2_line = eigarray(2, **kwargs)


q_full = np.linspace(-2, 2, 200)
#q2 = np.linspace(-2, 2, 100)
qx_full, qy_full = np.meshgrid(q_full, q_full)
kwargs = {'qx':qx_full, 'qy': qy_full, 'theta_rot':theta_rot, 
          'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3}
e0_full = eigarray(0, **kwargs)
e2_full = eigarray(2, **kwargs)
e1_full = eigarray(1, **kwargs)

#%%

#plt.figure(figsize=(3,3))
#plt.contourf(qx_full, qy_full, e0_full.T, cmap='viridis',
#                   levels=np.linspace(e0_full.min()*1.15, e0_full.max(), 10))
#plt.contour(qx_full, qy_full, e0_full.T, linewidths=1.,
#                levels=np.linspace(e0_full.min()*1.15, e0_full.max(), 10),
#                colors='White',linestyles='-')
#plt.xlabel('$q_x$, in units of $k_L$')
#plt.ylabel('$q_y$, in units of $k_L$')
#plt.xticks([-2,-1,0,1,2])
#plt.yticks([-2,-1,0,1,2])
#plt.tight_layout()
#plt.savefig('contour_rashba.pdf')
#plt.show()


#%%


q_full = np.linspace(-1.5, 1.5, 200)
#q2 = np.linspace(-2, 2, 100)
qx_full, qy_full = np.meshgrid(q_full, q_full)
kwargs = {'qx':qx_full, 'qy': qy_full, 'theta_rot':theta_rot, 
          'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3, 'regular':True}
e0_full = eigarray(0, **kwargs)
#e2_full = eigarray(2, **kwargs)
#e1_full = eigarray(1, **kwargs)

nn = 2.5
kwargs = {'qx':qx_full, 'qy': qy_full, 'theta_rot':theta_rot, 
          'Omega1':Omega*nn, 'Omega2':Omega2*nn, 'Omega3':Omega3*nn, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3, 'regular':True}
e0_big = eigarray(0, **kwargs)
#e2_full = eigarray(2, **kwargs)
#e1_full = eigarray(1, **kwargs)
nn = 100
kwargs = {'qx':qx_full, 'qy': qy_full, 'theta_rot':theta_rot, 
          'Omega1':Omega*nn, 'Omega2':Omega2*nn, 'Omega3':Omega3*nn, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3, 'regular':True}
e0_biggest = eigarray(0, **kwargs)
#%%
fig = plt.figure(figsize=(9./1.7,3./1.7))
#    
gs = GridSpec(1, 3)   
ax = plt.subplot(gs[0])
#plt.contourf(qx_full, qy_full, e0_full.T, cmap='viridis',
#                   levels=np.linspace(e0_full.min()*1.15, e0_full.max(), 15))
plt.pcolormesh(qx_full, qy_full, e0_full.T, rasterized=True)
plt.contour(qx_full, qy_full, e0_full.T, linewidths=1.,
                levels=np.linspace(e0_full.min()*1.25, e0_full.max(), 10),
                colors='White',linestyles='-')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
plt.ylabel('$q_y$, $\mathrm{in\ units\ of\ } k_L$')
plt.xticks([-1.5,0,1.5])
plt.yticks([-1.5,0,1.5])
ax.set_aspect('equal')

ax = plt.subplot(gs[1])
plt.pcolormesh(qx_full, qy_full, e0_big.T, rasterized=True)
plt.contour(qx_full, qy_full, e0_big.T, linewidths=1.,
                levels=np.linspace(e0_big.min()*1.18, e0_big.max(), 10),
                colors='White',linestyles='-')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
#plt.ylabel('$q_y$, in units of $k_L$')
plt.xticks([-1.5,0,1.5])
plt.yticks([-1.5,0,1.5])
ax.set_yticklabels([])
ax.set_aspect('equal')

ax = plt.subplot(gs[2])
plt.pcolormesh(qx_full, qy_full, e0_biggest.T, rasterized=True)
plt.contour(qx_full, qy_full, e0_biggest.T, linewidths=1.,
                levels=np.linspace(e0_biggest.min()*1.001, e0_biggest.max(), 10),
                colors='White',linestyles='-')

plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
plt.xticks([-1.5,0,1.5])
plt.yticks([-1.5,0,1.5])
ax.set_yticklabels([])
ax.set_aspect('equal')
plt.tight_layout()


plt.savefig('rashba_alien.pdf')
plt.show()
