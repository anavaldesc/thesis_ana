#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:53:44 2019

@author: banano
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:30:52 2019

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
#%%

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

#%%
def H_RashbaRF(qx, qy, 
               Omega1, Omega2, Omega3, 
               delta1, delta2, delta3,
               theta1=87.04, theta2=322.25, theta3=226.69):
    
#    theta2 = 0
#    theta3 = 120
#    theta1 = 240
    
#    k1_x = np.cos( 0.5 * np.pi + theta_rot) 
#    k1_y = -np.sin(0.5 * np.pi + theta_rot) 
#    k2_x = np.cos(1.75 * np.pi + theta_rot) 
#    k2_y = -np.sin(1.75 * np.pi + theta_rot) 
#    k3_x = np.cos(1.25 * np.pi + theta_rot)
#    k3_y = -np.sin(1.25 * np.pi + theta_rot)
    
    k1_x = np.cos(theta1 * np.pi / 180) 
    k1_y = np.sin(theta1 * np.pi / 180) 
    k2_x = np.cos(theta2 * np.pi / 180) 
    k2_y = np.sin(theta2 * np.pi / 180) 
    k3_x = np.cos(theta3 * np.pi / 180)
    k3_y = np.sin(theta3 * np.pi / 180)
#    
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 , Omega1, Omega3], 
          [Omega1, (qx+k2_x)**2 + (qy+k2_y)**2+delta2, Omega2], 
          [Omega3, Omega2, (qx+k3_x)**2 + (qy+k3_y)**2+delta3]])
    H = np.array(H)

    return H 


k = np.linspace(-1, 1, 2**6)

def energies(idx, **kwargs):

    return eigvalsh(H_RashbaRF(**kwargs))[idx]
eigarray = np.vectorize(energies)

q = np.linspace(-1, 1, 100)
q2 = np.linspace(-2, 0, 100)
qx, qy = np.meshgrid(q, q2)
Omega = 1.45*0.1
Omega1 = 1.5#1.72#Omega#11.8/(2*np.pi*np.sqrt(2))# Omega
Omega2 = 1.5#Omega#12/(2*np.pi*np.sqrt(2))#Omega
Omega3 = 1.5#Omega#11.3/(2*np.pi*np.sqrt(2))#Omega
sign = 0
delta1= 5*sign
delta2= 0
delta3= 5*sign
x=np.pi*0 #- 7*2*np.pi/360
y0 = 0
x0 = 0
#
#[11.8, 12, 11.3]
kwargs = {'Omega1':Omega1, 'Omega2':Omega2, 'Omega3':Omega3, 
          'delta1':delta1, 'delta2':delta2, 'delta3':delta3}

e_three = []


kwargs['qx'] = q*np.cos(x)-x0
kwargs['qy'] = q*np.sin(x)-y0 #+ raf.pix_to_k(1, 0,bin_image=False)/k_L*(100-i)

for j in range(3):
    ee = eigarray(j, **kwargs)
#    plt.plot(q, ee, color='k', linewidth=2)
    e_three.append(ee)
    
#plt.savefig('figures/bands_neg_delta.pdf')
#plt.show()
#
##%%

#%%
n_manifolds = 5

def block_matrix(n_bands, n_block):
    block = np.eye(n_bands+n_block)[n_block::,0:-n_block]
    return block + block.T

def Rashba_floquet(kx, ky, n_manifolds, omega_fl, Omega_zx, Omega_xy, Omega_zy,
                   off_resonant=False):

    kwargs = {'Omega1':0, 'Omega2':0, 'Omega3':0, 
              'delta1':0, 'delta2':0, 'delta3':0}
    kwargs['qx'] = kx
    kwargs['qy'] = ky
    
    n_bands = 2*n_manifolds + 1
    
    floquet_n = np.arange(-n_manifolds, n_manifolds+1, 1)
    floquet_diag = np.kron(np.diag(floquet_n)*omega_fl, np.eye(3))
    floquet_diag += np.kron(np.eye(n_bands), H_RashbaRF(**kwargs)+np.diag([0, 2*omega_fl, 3*omega_fl]))
     
    f_xy = np.array([[0, 0, 0], [0, 0, 1],[0, 1, 0]]) * Omega_xy * 1
    f_zx = np.array([[0, 1, 0], [1, 0, 0],[0, 0, 0]]) * Omega_zx * 1
    f_zy = np.array([[0, 0, 1], [0, 0, 0],[1, 0, 0]]) * Omega_zy * 1
    
    mat_zx = np.kron(block_matrix(n_bands, 2), f_zx)
    mat_zy = np.kron(block_matrix(n_bands, 3), f_zy)
    mat_xy = np.kron(block_matrix(n_bands, 1), f_xy)
    
    mat_zx_off = np.kron(block_matrix(n_bands, 3), f_zx) / 1.5
    mat_zy_off = np.kron(block_matrix(n_bands, 2), f_zy) / 1.5
    
    
    H_floquet = floquet_diag + mat_zx + mat_xy + mat_zy 
    if off_resonant:
        H_floquet += mat_zx_off + mat_zy_off
    
    return H_floquet

def floquet_energies(idx, **kwargs):

    return eigvalsh(Rashba_floquet(**kwargs))[idx]

floquet_eigenarray = np.vectorize(floquet_energies)

Om = 1.5
kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
          'Omega_zy':Om, 'omega_fl':1*83.24/3.678}


e_floquet = []

for i in range(16, 19):
    ee = floquet_eigenarray(i, **kwargs)
#    plt.plot(ee, 'k')
    e_floquet.append(ee)
    
e_floquet_off = []

kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
          'Omega_zy':Om, 'omega_fl':1*83.24/3.678, 'off_resonant':True}

e_floquet_large = []
for i in range(16, 19):
    ee = floquet_eigenarray(i, **kwargs)
#    plt.plot(ee, 'k')
    e_floquet_off.append(ee)

kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
          'Omega_zy':Om, 'omega_fl':5*83.24/3.678}


for i in range(16, 19):
    ee = floquet_eigenarray(i, **kwargs)
#    plt.plot(ee, 'k')
    e_floquet_large.append(ee)
#%%

fig = plt.figure(figsize=(7/1.7,4./1.7))
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0])    
    
e_floquet = np.array(e_floquet)
e_floquet -= e_floquet.min()
e_floquet_large = np.array(e_floquet_large)
e_floquet_large -= e_floquet_large.min()
e_three = np.array(e_three)
e_three -= e_three.min()
plt.plot(q, e_three.T, 'k-', label='effective Hamiltonian')
plt.plot(q, e_floquet.T, 'k--', label='Floquet Hamiltonian')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
plt.ylabel('$\mathrm{Energy\ in\ units\ of\ } E_{\mathrm{L}}$')
#plt.legend(loc='upper center')
plt.ylim([0,1])
plt.yticks([0,0.5,1])
ax = plt.subplot(gs[1])   

plt.plot(q, e_three.T, 'k-', label='effective Hamiltonian')
plt.plot(q, e_floquet_large.T, 'k--', label='Floquet Hamiltonian')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
#plt.ylabel('Energy in units of $E_L$')
#plt.legend(loc='upper right')
plt.ylim([0,1]) 
ax.set_yticklabels([])
plt.tight_layout()
plt.savefig('floquet_effects.pdf')

#%%

fig = plt.figure(figsize=(7/1.7,4./1.7))
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0])    
    
e_floquet = np.array(e_floquet)
e_floquet -= e_floquet.min()
e_floquet_large = np.array(e_floquet_large)
e_floquet_large -= e_floquet_large.min()
e_floquet_off = np.array(e_floquet_off)
e_floquet_off -= e_floquet_off.min()
e_three = np.array(e_three)
e_three -= e_three.min()
plt.plot(q, e_three.T, 'k-', label='effective Hamiltonian')
plt.plot(q, e_floquet.T, 'k--', label='Floquet Hamiltonian')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
plt.ylabel('$\mathrm{Energy\ in\ units\ of\ } E_{\mathrm{L}}$')
#plt.legend(loc='upper center')
plt.ylim([0,1])
plt.yticks([0,0.5,1])
ax = plt.subplot(gs[1])   

plt.plot(q, e_three.T, 'k-', label='effective Hamiltonian')
plt.plot(q, e_floquet_off.T, 'k--', label='Floquet Hamiltonian')
plt.xlabel('$q_x$, $\mathrm{in\ units\ of\ } k_L$')
#plt.ylabel('Energy in units of $E_L$')
#plt.legend(loc='upper right')
plt.ylim([0,1]) 
ax.set_yticklabels([])
plt.tight_layout()
plt.savefig('floquet_effects.pdf')


#%%

om_fl_vec = np.linspace(40/3.678, 5*83.24/3.67, 20)
diff = []
for om_fl in om_fl_vec:
    
    e_floquet_large = []

    kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
              'Omega_zy':Om, 'omega_fl':om_fl}

    for i in range(16, 19):
        ee = floquet_eigenarray(i, **kwargs)
        e_floquet_large.append(ee)

    e_floquet_large = np.array(e_floquet_large)
    e_floquet_large -= e_floquet_large.min()
    
    diff.append(np.abs(e_floquet_large-e_three).sum())
    
plt.plot(om_fl_vec*3.678, diff, 'o')
