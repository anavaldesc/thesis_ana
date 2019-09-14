# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:01:35 2018

@author: Ana
"""



import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
from matplotlib import rcParams
#%%

#fontsize = 9
#rcParams['axes.labelsize'] = fontsize
#rcParams['xtick.labelsize'] = fontsize
#rcParams['ytick.labelsize'] = fontsize
#rcParams['legend.fontsize'] = fontsize
#
#rcParams['pdf.fonttype'] = 42 # True type fonts
##rcParams['font.family'] = 'sans-serif'
##rcParams['font.family'] = 'serif'
##rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True
#rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
##rcParams['text.latex.preamble'] = [
##       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
##       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
##       r'\usepackage{helvet}',    # set the normal font here
##       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
##       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
##]  
#
#rcParams['axes.linewidth'] = 0.75
#rcParams['lines.linewidth'] = 0.75
#
#rcParams['xtick.major.size'] = 3      # major tick size in points
#rcParams['xtick.minor.size'] = 2      # minor tick size in points
#rcParams['xtick.major.width'] = 0.75       # major tick width in points
#rcParams['xtick.minor.width'] = 0.75      # minor tick width in points
#
#rcParams['ytick.major.size'] = 3      # major tick size in points
#rcParams['ytick.minor.size'] = 2      # minor tick size in points
#rcParams['ytick.major.width'] = 0.75       # major tick width in points
#rcParams['ytick.minor.width'] = 0.75      # minor tick width in points

def H_Rashba(t, qx, qy, Delta_z):
    
    sigma_x = np.array([[0, 1], [1, 0]], dtype='complex')
    sigma_y = np.array([[0, -1j], [-1j, 0]], dtype='complex')
    sigma_z = np.array([[1, 0], [0, -1]], dtype='complex')
    sigma_0 = np.array([[1, 0], [0, 1]], dtype='complex')
    
    H = (qx**2 + qy**2) * sigma_0
    H += sigma_x * qy - sigma_y * qx + Delta_z * sigma_z 
    
    return H

def H_RashbaRF(t, qx, qy, Omega1, Omega2, Omega3):
    
    """
    Simple Rashba model from Dans paper
    """
    
    Omega1 = Omega1 * 2 * np.pi
    Omega2 = Omega2 * 2 * np.pi
    Omega3 = Omega3 * 2 * np.pi
    
    k2_x = np.cos(2 * np.pi / (360./135))
    k2_y = -np.sin(2 * np.pi / (360./135))
    k1_x = np.cos(2 * np.pi  / 1.6)
    k1_y = -np.sin(2 * np.pi / 1.6)
    k3_x = np.cos(2 * np.pi)
    k3_y = -np.sin(2 * np.pi)
    
    k2_x = np.cos(2 * np.pi *1/3)
    k2_y = -np.sin(2 * np.pi *1/3)
    k3_x = np.cos(2 * np.pi  *2/ 3)
    k3_y =-np.sin(2 * np.pi *2/ 3)
    k1_x = np.cos(2 * np.pi)
    k1_y = -np.sin(2 * np.pi)
    
    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2, 1.0j*Omega1, Omega3], 
          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2, -1.0j*Omega2], 
          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2]])
    H = np.array(H, dtype='complex')

    return H 

#
#def H_RashbaRF(qx, qy, 
#               Omega1, Omega2, Omega3, 
#               delta1=0, delta2=0, delta3=0,
#               theta1=91.04, theta2=317.95, theta3=227.64):
#    
#    k1_x = np.cos(theta1 * np.pi / 180) 
#    k1_y = np.sin(theta1 * np.pi / 180) 
#    k2_x = np.cos(theta2 * np.pi / 180) 
#    k2_y = np.sin(theta2 * np.pi / 180) 
#    k3_x = np.cos(theta3 * np.pi / 180)
#    k3_y = np.sin(theta3 * np.pi / 180)
##    
#    
#    H = np.array([[(qx+k1_x)**2 + (qy+k1_y)**2+delta1 , 1.0j*Omega1, Omega3], 
#          [-1.0j*Omega1, (qx+k2_x)**2 + (qy+k2_y)**2+delta2, -1.0j*Omega2], 
#          [Omega3, 1.0j*Omega2, (qx+k3_x)**2 + (qy+k3_y)**2+delta3]])
#    H = np.array(H, dtype='complex')
#
#    return H 

initial_state = 0
k_dim = 2**6
k_min = 1.5
kvec  = np.linspace(-k_min, k_min, k_dim) 
Er = 3.678 * 4
Omega_Raman = 1.45 #/ Er 
Omega1 = 1.739
Omega2 = 1.18
Omega3 = 1.31

Omega1_load = Omega1 / Er 
Omega2_load = Omega2 / Er 
Omega3_load = Omega3 / Er 
Delta_z = 0.0
qx = 0
qy = 0

make_psi_array = True
RashbaRF = True
Topology = True


if RashbaRF:
    H_dim = 3
else:
    H_dim = 2
    
Psi_arr = np.empty((k_dim, k_dim, H_dim), dtype=complex)
eigen_arr = np.empty((k_dim, k_dim, H_dim))
#phase_gauge = np.empty([kdim, kdim])
if RashbaRF:
    args_load = [qx, qy,  Omega1_load,  Omega2_load, Omega3_load]
else:
    args_load = [qx, qy, Delta_z]

if make_psi_array:
    
    for i, kx in tqdm.tqdm(enumerate(kvec)):
        for j, ky in enumerate(kvec):
    
            args_load[0] = kx
            args_load[1] = ky
            if RashbaRF:
                Psi0 = eigh(H_RashbaRF(0, *args_load))[1][:,initial_state]
                eigen_arr[i, j] = eigh(H_RashbaRF(0, *args_load))[0]
            else:
                Psi0 = eigh(H_Rashba(0, *args_load))[1][:,initial_state]
                eigen_arr[i, j] = eigh(H_Rashba(0, *args_load))[0]
            Psi_arr[i, j] = Psi0
            
            phase_gauge = np.angle((i-k_dim/2)+(j-k_dim/2)*1.0j)
            
    Psi_real = np.abs(Psi_arr)
    Psi_phase = (np.angle(Psi_arr)+np.pi)#%2*np.pi

if RashbaRF:
    for i in range(3):
        Psi_phase[:,:,i] += Psi_phase[:,:,2].max()
        Psi_phase[:,:,i] -= Psi_phase[:,:,2]
        Psi_phase[:,:,i] = Psi_phase[:,:,i]%(2*np.pi)
    
    for i, kx in tqdm.tqdm(enumerate(kvec)):
        for j, ky in enumerate(kvec):
            if np.isclose(Psi_phase[i,j,0]/np.pi, 0):
#                print('here')
                Psi_phase[i,j,0] = 2*np.pi
    for i in [2, 1, 0]:
        Psi_phase[:,:,i] = (Psi_phase[:,:,i]-Psi_phase[:,:,2])%(2*np.pi)
        
#    Psi_phase[:,:,1] += (Psi_real[:,:,0]*np.pi)%(2*np.pi)
#    Psi_phase = np.exp(1.0j*Psi_phase)
#    for i in range(3):
#        
#        Psi_phase[:,:,i] *= np.exp(-1.0j*np.angle(Psi_phase[:,:,1]))
#    Psi_phase = np.log(Psi_phase).imag
#            

else:
    
    for i in range(2):
        Psi_phase[:,:,i] -= Psi_phase[:,:,1]
    for i, kx in tqdm.tqdm(enumerate(kvec)):
        for j, ky in enumerate(kvec):
            if Psi_phase[i,j,0] < 0:
    #            print('here')
                Psi_phase[i,j,0] += 2*np.pi


Psi_phase = np.log(np.exp(1.0j*Psi_phase)).imag

#plt.imshow(eigen_arr[:,:,1]-eigen_arr[:,:,0], vmin=0)
#plt.imshow(eigen_arr[:,:,0])
#plt.colorbar(label='ground state energy in units of EL')
#plt.show() 
#plt.plot(kvec, eigen_arr[:,int(k_dim/2), 0:2], 'k')
#plt.xlabel('qx in units of kL')
#eigen_diff = eigen_arr[:,:,1]-eigen_arr[:,:,0]
#idx = np.argmin(eigen_diff.ravel())  
#%% 

scale = 0.8
x, y = Psi_phase[:,:,0].shape
x_vec = np.linspace(-int(x/2), int(x/2), x)
y_vec = np.linspace(-int(y/2), int(y/2), y)

x_arr, y_arr = np.meshgrid(kvec, kvec)

plt.figure(figsize=(2.3*H_dim*scale,4*scale))
gs = GridSpec(2, H_dim)
factor = 2
rep = 0

for i in range(H_dim):
    ax = plt.subplot(gs[0,i])
    ax.set_aspect('equal')
#    plt.title('Phi%s'%(i+1))
#    plt.title(r'$\phi_%s$'%(i+1))
    plt.pcolormesh(x_arr, y_arr, 
                   np.rot90(Psi_phase[:,:,i],3)/np.pi, 
                   vmin=-1, vmax=1, 
                   cmap='hsv', rasterized=True)
    ax.set_xticklabels([])
    if i == 0:
        plt.ylabel('$q_y\ \mathrm{in\ units\ of\ } k_L$')
    else:
        ax.set_yticklabels([])
    if i == 2:
        plt.colorbar(shrink=0.95, ticks=[-1, 0, 1], 
                     label=r'$\mathrm{phase\ in\ units\ of\ } \pi$')

    ax = plt.subplot(gs[1,i])
    ax.set_aspect('equal')
#    plt.title('Phi%s'%(i+1))
#    plt.title(r'$\vert a_{1,%s}\vert ^2$'%(i+1))
    plt.pcolormesh(x_arr, y_arr, 
                   np.rot90((Psi_real[:,:,i])**2,3), 
                   vmin=0, vmax=1,
                   rasterized=True)
    plt.xlabel('$q_x\ \mathrm{in\ units\ of\ } k_L$')
    if i == 0:
        plt.ylabel('$q_y\ \mathrm{in\ units\ of\ } k_L$')
    else:
        ax.set_yticklabels([])
    if i == 2:
        plt.colorbar(shrink=0.95, ticks=[0, 0.5,  1], label=r'$\mathrm{amplitude}$')

plt.tight_layout()
#plt.savefig('nontopological_eigenvecs.pdf')

plt.show()   

#%%

plt.figure(figsize=(2.3*H_dim*scale,4*scale))
gs = GridSpec(2, H_dim)
factor = 2
rep = 0

for i in range(H_dim):
    ax = plt.subplot(gs[0,i])
    ax.set_aspect('equal')
#    plt.title('Phi%s'%(i+1))
#    plt.title(r'$\phi_%s$'%(i+1))
    phase_diff = -Psi_phase[:,:,i] + Psi_phase[:,:,(i+1)%3]
    phase_diff = np.log(np.exp(1.0j*phase_diff)).imag
    plt.pcolormesh(x_arr, y_arr, 
                   np.rot90(phase_diff,2)/np.pi, 
                   vmin=-1, vmax=1, 
                   cmap='hsv', rasterized=True)
    ax.set_xticklabels([])
    if i == 0:
        plt.ylabel('$q_y\ \mathrm{in\ units\ of\ } k_L$')
    else:
        ax.set_yticklabels([])
    if i == 2:
        plt.colorbar(shrink=0.95, ticks=[-1, 0, 1], 
                     label=r'$\mathrm{phase\ in\ units\ of\ } \pi$')

    ax = plt.subplot(gs[1,i])
    ax.set_aspect('equal')
#    plt.title('Phi%s'%(i+1))
#    plt.title(r'$\vert a_{1,%s}\vert ^2$'%(i+1))
    plt.pcolormesh(x_arr, y_arr, 
                   np.rot90((Psi_real[:,:,i])**2,2), 
                   vmin=0, vmax=1,
                   rasterized=True)
    plt.xlabel('$q_x\ \mathrm{in\ units\ of\ } k_L$')
    if i == 0:
        plt.ylabel('$q_y\ \mathrm{in\ units\ of\ } k_L$')
    else:
        ax.set_yticklabels([])
    if i == 2:
        plt.colorbar(shrink=0.95, ticks=[0, 0.5,  1], label=r'$\mathrm{amplitude}$')

plt.tight_layout()