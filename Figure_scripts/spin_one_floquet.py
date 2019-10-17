#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:38:19 2019

@author: banano
"""

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

fontsize = 7
rcParams['axes.labelsize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize

rcParams['pdf.fonttype'] = 42 # True type fonts
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{cmbright}', r'\usepackage{amsmath}']
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
def H_0(q, delta, epsilon):
    
    H = np.array([[(q-2)**2 , 0, 0], 
          [0, q**2-epsilon, 0], 
          [0, 0, (q+2)**2]])
    H = np.array(H)

    return H 

def H_SOC(q, delta, epsilon, omega):
    
    omega /= np.sqrt(2)
    H = np.array([[(q-2)**2 , omega, 0], 
          [omega, q**2-epsilon, omega], 
          [0, omega, (q+2)**2]])
    H = np.array(H)

    return H 

k = np.linspace(-3, 3, 2**6)

def energies(idx, **kwargs):

    return eigvalsh(H_SOC(**kwargs))[idx]
eigarray = np.vectorize(energies)

q = np.linspace(-1, 1, 100)
q2 = np.linspace(-2, 0, 100)
qx, qy = np.meshgrid(q, q2)
Omega = 1.45
delta = 0
epsilon=0
#

fig = plt.figure(figsize=(2.2,2.7))
gs = GridSpec(2, 1)   
ax = plt.subplot(gs[0])    
#[11.8, 12, 11.3]
kwargs = {'omega':Omega, 'delta':0, 'epsilon':0.1, 'q':k}
e = eigarray(0, **kwargs)
plt.plot(k, e-e[2**5]*0, color='teal', linewidth=1)
kwargs['epsilon'] = -0.5
e = eigarray(0, **kwargs)
plt.plot(k, e-e[2**5]*0, color='darkslategray', linewidth=1)
ax.set_xticklabels([])


ax = plt.subplot(gs[1])

kwargs = {'omega':5.5, 'delta':0, 'epsilon':0, 'q':k}
plt.plot(k, eigarray(0, **kwargs), color='teal', linewidth=1)
kwargs['epsilon'] = -4
plt.plot(k, eigarray(0, **kwargs), color='darkslategray', linewidth=1)
plt.xlabel('$q$, $\mathrm{in\ units\ of\ } k_{\mathrm{L}}$')
plt.tight_layout()
plt.savefig('phase_transitions.pdf', transparent=True)
#%%
n_manifolds = 8

def block_matrix(n_bands, n_block):
    block = np.eye(n_bands+n_block)[n_block::,0:-n_block]
    return block + block.T

def SOC_floquet(k, n_manifolds, omega_fl, Omega, epsilon):

    kwargs = {'epsilon':epsilon, 'delta':0, 'q':k}
    
    n_bands = 2*n_manifolds + 1    
    floquet_n = np.arange(-n_manifolds, n_manifolds+1, 1)
    floquet_diag = np.kron(np.diag(floquet_n)*omega_fl, np.eye(3))
    floquet_diag += np.kron(np.eye(n_bands), H_0(**kwargs))
    
#    plt.imshow(floquet_diag)
#    plt.colorbar()
    
    f_x = 1/np.sqrt(2)* np.array([[0, 1, 0],
                                  [1, 0, 1],
                                  [0, 1, 0]])
    
    
    mat_x = np.kron(block_matrix(n_bands, 1), f_x*Omega)
    
    

    H_floquet = floquet_diag + mat_x
    
    return H_floquet

def floquet_energies(idx, **kwargs):

    return eigvalsh(SOC_floquet(**kwargs))[idx]

floquet_eigenarray = np.vectorize(floquet_energies)

Om = 2.2
eps = 46
kwargs = {'k':k, 'n_manifolds':5, 'Omega':Om, 'epsilon':eps/3.678, 'omega_fl':eps/3.678}


e_floquet = []

for i in range(19-10, 20):
    ee = floquet_eigenarray(i, **kwargs)
#    plt.plot(ee, 'k')
    e_floquet.append(ee)
    
#e_floquet_off = []
#
#kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
#          'Omega_zy':Om, 'omega_fl':1*83.24/3.678, 'off_resonant':True}
#
#e_floquet_large = []
#for i in range(16, 19):
#    ee = floquet_eigenarray(i, **kwargs)
##    plt.plot(ee, 'k')
#    e_floquet_off.append(ee)
#
#kwargs = {'kx':q, 'ky':0, 'n_manifolds':6, 'Omega_zx':Om, 'Omega_xy':Om, 
#          'Omega_zy':Om, 'omega_fl':5*83.24/3.678}
#
#
#for i in range(16, 19):
#    ee = floquet_eigenarray(i, **kwargs)
##    plt.plot(ee, 'k')
#    e_floquet_large.append(ee)
#%%

fig = plt.figure(figsize=(5.6,2.5))
gs = GridSpec(1, 2)   

ax = plt.subplot(gs[0])


kwargs = {'omega':Omega/2, 'delta':0, 'epsilon':0, 'q':k}
for i in range(3):
    plt.plot(k, eigarray(i, **kwargs), color='darkslategray', linewidth=1)
plt.xlabel('$q$, $\mathrm{in\ units\ of\ } k_{\mathrm{L}}$')


plt.xlim([-3, 3])
ax.set_xticks([-2, 0, 2])
plt.ylim([-0.5, 25])

plt.ylabel('$\mathrm{Energy\ in\ units\ of\ } E_{\mathrm{L}}$')
ax = plt.subplot(gs[1])    
    
e_floquet = np.array(e_floquet)
#e_floquet -= e_floquet.min()
#e_floquet_large = np.array(e_floquet_large)
#e_floquet_large -= e_floquet_large.min()
#e_three = np.array(e_three)
#e_three -= e_three.min()
#plt.plot(q, e_three.T, 'k-', label='effective Hamiltonian')
plt.plot(k, e_floquet.T, '-', color='darkslategray', label='Floquet Hamiltonian',
         linewidth=1.)
plt.xlabel('$q$, $\mathrm{in\ units\ of\ } k_{\mathrm{L}}$')
ax.set_xticks([-2, 0, 2])
plt.ylim([-25.5-1, 12-1])
plt.xlim([-3, 3])
plt.tight_layout()

plt.savefig('spin_one_floquet.pdf', transparent=True)
#plt.vlines(-1.3, -30, 12+1)