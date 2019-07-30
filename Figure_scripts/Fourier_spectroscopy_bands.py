# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:02:35 2016

@author: banano
"""

import sys
sys.path.append('/Users/banano/Documents/UMD/Research/MolmerSorensen/Analysis')
#import matplotlib as mpl
import scipy.fftpack as sf
import MSfit as TDSE
import MolmerFloquet as mf
import matplotlib.pyplot as plt
import numpy as np
from math import floor
import TDSEfit as tdse
#import databandit.functions as dbf
#mpl.rcParams['axes.color_cycle'] = ['r', 'k', 'c']


def generate_bands(vals, tstart, tstop, nt, nd):


    El = 1839
    Omega0 = vals['Omega0']*El
    Omega = vals['Omega']*El
    Omega2 = vals['deltaomg']*El
    delta = vals['delta']*El
    epsilon = vals['eps']*El
    phi = vals['phi']
    phi2 = vals['phi']
#    
    tt = np.linspace(tstart, tstop, nt)
    deltavec = np.linspace(-12*El, 12*El, nd)
    pp = np.linspace(np.min(deltavec)/(4 * El), np.max(deltavec)/(4 * El), 1e2)    
    
    sp = []

    for delta in deltavec:
        
        pops = TDSE.TDSE(tt,Omega0, Omega, Omega2 , delta, epsilon, phi, phi2)
        pops -= np.mean(pops, axis=0)
        N = len(pops)
        dt = tt[2]-tt[1]
        xf = np.linspace(0.0, 1/(2 * dt), N  / 2 )
        fpops = np.array([np.abs(sf.fft(pops[:, i])**2) for i in range(3)])    
        fpops = fpops/np.max(fpops)
        fpops = fpops[:, :int(N/2)].T
#        for i in range(0,3):
#            fpops[:,i] = fpops[:,i]/np.max(fpops[:,i])
#        fpops = np.mean(fpops, axis=1)
        sp.append(fpops)
#        sp.append(fpops[:,0]-fpops[:,2])
        
    sp = np.array(sp)
    deltavec = np.linspace(-12*El, 12*El, nd+1)
    grid = np.meshgrid(deltavec/El, xf*1e-3, sparse=False, indexing='ij')


    floquetbands = np.zeros([12,len(pp)])
    bandnum = 3*(2*vals['blocks']+1)
    mid = int(floor(bandnum/2))+1
    
    for i in range(mid-4,mid+8):
        
        floquetbands[i-(mid-4)] = mf.eigarray(vals, pp, i)
   
   
    dif1 = np.array(np.diff(floquetbands, n=1, axis=0))
    dif2 = np.array(np.diff(floquetbands[::2], n=1, axis=0))
    dif3 = np.array(np.diff(floquetbands[::3], n=1, axis=0))
    dif4 = np.array(np.diff(floquetbands[::4], n=1, axis=0))
    dif5 = np.array(np.diff(floquetbands[::5], n=1, axis=0))
    dif6 = np.array(np.diff(floquetbands[::6], n=1, axis=0))
    dif7 = np.array(np.diff(floquetbands[::7], n=1, axis=0))

    plt.figure(figsize=(5,4))
#    for i in range(0,8):
#        plt.plot(pp*4,(dif1[i]*1.839+0**2),'c--', linewidth = 1.5)
#    for i in range(0,4):
#        plt.plot(pp*4,(dif2[i]*1.839+0**2),'r--', linewidth = 1.5)
#    for i in range(0,2):
#        plt.plot(pp*4,(dif3[i]*1.839+0**2),'g', linewidth = 1.5) 
    
    plt.plot(pp*4, dif1.T*1.839,'c--', linewidth=2)
    plt.plot(pp*4, dif2.T*1.839,'r--',linewidth=2)
    plt.plot(pp*4, dif3.T*1.839,'g--', linewidth=2)  
    plt.plot(pp*4, (dif1.T+dif3.max())*1.839,'--', color='purple', linewidth=2)         
    plt.plot(pp*4, (dif2.T+dif3.max())*1.839,'--', color='orange', linewidth=2)
    plt.plot(pp*4, dif6.T*1.839,'g--', linewidth=2)  

    plt.pcolormesh(grid[0],grid[1],sp[:,:,1],cmap='Greys',vmin=0, vmax=0.20)
    plt.xlabel('detuning [$E_L$]')
    plt.ylabel('frequency [kHz]')
    plt.axis('tight')
    plt.ylim([0,2*dif3.max()*1.839+5])

    return (grid, sp, deltavec, pp, dif1, dif2, dif3, dif4, dif5, dif6)
    
    
##Define parameters for bands    
Omega  = 8.5
Omega0 = 0
eps = 15.46865104480917
deltaomg = -12
pp = np.linspace(-12,12,2**8)
nn = 15
tstart = 5e-6
tstop = 700e-6
nt = 180
nd = 25
phi = 0*np.pi

vals = {'Omega0':Omega0, 'Omega': Omega, 'deltaomg':deltaomg, 
        'eps':eps,'delta':0, 'phi': phi, 'blocks': nn}
        
#band = generate_bands(vals, tstart, tstop, nt, nd)
  #%%
#import matplotlib.colors as mcolors
from numpy.ma import masked_array

bp = band[1][:,:,0]
#bp = bp/bp.max()
bm = band[1][:,:,2]
bsum = np.abs(bp-bm)
#bm = bm/bm.max()
mbp = masked_array(bp, bp < 0.00075)
mbm = masked_array(bm, bm < 0.00075)
mbsum = bp + bm +band[1][:,:,1]- 0*np.abs(mbm-mbp)

plt.pcolormesh(band[0][0], band[0][1], mbsum, cmap='Greys', vmin=0, vmax=1)
#plt.pcolormesh(band[0][0], band[0][1], mbm-mbp, cmap='RdBu', vmin=-0.2, vmax=0.2)
#plt.pcolormesh(band[0][0], band[0][1], mbp, cmap='Reds', vmin=0.001, vmax=0.5)
#plt.pcolormesh(band[0][0], band[0][1], mbm, cmap='Blues', vmin=0.001, vmax=0.5)

#plt.ylim([0,135])
#pp = np.linspace(-12+3.5,9+3.5,len(band[3][0]))
#plt.xlim([0,12])7
#plt.ylim([0,70])
plt.xlim([-12,12])
#plt.axis('Tight')
plt.show()

#
#from numpy.ma import masked_array
#
#v1 = -1+2*np.random.rand(50,150)
#v1a = masked_array(v1,v1<0)
#v1b = masked_array(v1,v1>=0)
#fig,ax = plt.subplots()
#pa = ax.imshow(v1a,interpolation='nearest',cmap=cm.Reds)
#cba = plt.colorbar(pa,shrink=0.25)
#pb = ax.imshow(v1b,interpolation='nearest',cmap=cm.winter)
#cbb = plt.colorbar(pb,shrink=0.25)
#plt.xlabel('Day')
#plt.ylabel('Depth')
#cba.set_label('positive')
#cbb.set_label('negative')
#plt.show()
#%%
vals['Omega0'] = 1
vals['Omega'] = 8.5
vals['deltaomg'] = -18
vals['eps'] = 15.46865104480917
pp = np.linspace(-3,3,100)
#for i in range(45,48):
#    plt.plot(pp, mf.eigarray(vals, pp, i) - mf.eigarray(vals, 0, 46),'b')
#plt.show()
gaps = []
Omega0vec = np.linspace(0,15,100)
for Omega0 in Omega0vec:
    
    vals['Omega0'] = Omega0
    gap = mf.eigarray(vals, 0, 47) -mf.eigarray(vals, 0, 46)
    gaps.append(gap)

plt.plot(Omega0vec,gaps,'b')


vals['Omega0'] = 10.7
vals['Omega'] = 7
vals['deltaomg'] = -17
vals['eps'] = 22.839
pp = np.linspace(-3,3,100)


gaps2 = []
Omega0vec = np.linspace(0,15,30)
for Omega0 in Omega0vec:
    
    vals['Omega0'] = Omega0
    gap = mf.eigarray(vals, 0, 47) -mf.eigarray(vals, 0, 46)
    gaps2.append(gap)

plt.plot(Omega0vec,gaps2,'k')

#%%
vals['blocks'] = 15
vals['Omega'] = 8.5
pp = np.linspace(-2.5,2.5,100)
vals['deltaomg'] = -12
vals['Omega0'] = 0

pp = np.linspace(-3,3,100)

for i in range(9*3,9*3+3):
    plt.plot(pp, mf.eigarray(vals, pp, i) - mf.eigarray(vals, 0, 27),'r')
plt.axis('Tight')

pfit = np.linspace(-0.5,0.5,100)
poli = np.polyfit(pfit, mf.eigarray(vals, pfit, 27),2)

vals['blocks'] = 15
vals['Omega'] = 0
#pp = np.linspace(-2.5,2.5,100)
vals['deltaomg'] = -18
vals['Omega0'] = 0


for i in range(9*3,9*3+3):
    plt.plot(pp, mf.eigarray(vals, pp, i) - mf.eigarray(vals, 0, 27),'k')
plt.axis('Tight')

poli2 = np.polyfit(pfit, mf.eigarray(vals, pfit, 27),2)

print(poli[0]/poli2[0])

#plt.show()
#%%
vals['Omega0'] = 8.5
vals['Omega'] = 5
vals['epsilon'] = 25
for i in range(46,48):
    plt.plot(pp, mf.eigarray(vals, pp, i) - mf.eigarray(vals, 0, 46),'k')  
plt.axis('Tight')
 #%%
from numpy.linalg import eigvalsh
import matplotlib.colors as colors
import matplotlib.cm as cmx

def HR(p,om): 

    H = [[(p-1)**2,om],[om,(p+1)**2]];
    return H
    
def eigvals(p,om,n):
    
    a = np.sort(eigvalsh(HR(p,om)));
    return a[n]

eigarray = np.vectorize(eigvals)

pp = np.linspace(-2.5,2.5,1e2)
pd = np.linspace(-2,2,1e3)
om_vec = np.linspace(0,1.5,6)
plt.figure(figsize=(7,5))
jet = cm = plt.get_cmap('viridis') 
cNorm  = colors.Normalize(vmin=0, vmax=1.7)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

plt.plot(pp, eigarray(pp,om_vec[0],1),pp, eigarray(pp,om_vec[0],0),'--', color='k')
for i in range(1,len(om_vec)):
    colorVal = scalarMap.to_rgba(om_vec[i])
    plt.plot(pp, eigarray(pp,om_vec[i],1),pp, eigarray(pp,om_vec[i],0), 
             color=colorVal, linewidth = 2)
             
plt.ylabel('$\mathrm{Energy  \ \ E/E_r}$',fontsize = 14)    
plt.xlabel('$\mathrm{Quasimomentum \ \ k/k_r}$', fontsize = 14)
plt.xlim([-2.5,2.5])
plt.ylim([-1, 10])
plt.show()
#%%
from numpy.linalg import eigvalsh
import matplotlib.colors as colors
import matplotlib.cm as cmx

def HR3(p,om, delta, eps): 
    om = om/np.sqrt(2)
    H = [[(p-2)**2+delta,om,0],[om,p**2-eps,om],[0,om,(p+2)**2-delta]];
    return H
    
def eigvals(p,om, delta, eps, n):
    
    a = np.sort(eigvalsh(HR3(p,om, delta, eps)));
    return a[n]

eigarray = np.vectorize(eigvals)

pp = np.linspace(-3,3,1e3)
pd = np.linspace(-2,2,1e3)
om_vec = np.linspace(0.0,0.0,2)
plt.figure(figsize=(1.02,1.34))
jet = cm = plt.get_cmap('viridis') 
cNorm  = colors.Normalize(vmin=0, vmax=7.5)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

delta = 0
eps = 15.429
om = 7
#%%

delta=8
gs = gridspec.GridSpec(1,1)
x= 1.3
pp = np.linspace(-2.5,2.5,1e2)
plt.figure(figsize=(1.02*x,1.1*x))
gs.update(left=0.4, right=0.95, bottom=0.25, top=0.95, wspace=1.7, hspace=0.15)  
#plt.ylabel('PSD [au]')
plt.subplot(gs[0])
#plt.plot(pp, eigarray(pp,om_vec[0],1),pp, eigarray(pp,om_vec[0],0),'--', color='k')
for i in range(1,len(om_vec)):
    colorVal = scalarMap.to_rgba(om_vec[i])
    plt.plot(pp, eigarray(pp,om, delta, eps, 1)-eigarray(0,om, delta, eps, 0),
             pp,  eigarray(pp,om, delta, eps, 2)-eigarray(0,om, delta, eps, 0),
             pp,  eigarray(pp,om, delta, eps, 0)-eigarray(0,om, delta, eps, 0),
             color='black', linewidth = 2)
min_k = np.argmin(eigarray(pp,om, delta, eps, 0)-eigarray(0,om, delta, eps, 0))
p_min = pp[min_k]
plt.plot(p_min, eigarray(p_min,om, delta, eps, 0)-eigarray(0,om, delta, eps, 0),'ko',)            
plt.ylabel('$E/E_{\mathrm{L}}$',fontsize = 8)    
plt.xlabel('$q_x/k_{\mathrm{L}}$', fontsize = 8)
plt.xlim([-3.,3.])
plt.xticks([-3, 3, 0], fontsize=8)
plt.yticks([0, 20, 40], fontsize=8)
#plt.ylim([-20, 30])
#plt.savefig('/Users/banano/Documents/UMD/Research/MolmerSorensen/Paper/Figures/dispersion2.pdf')
#plt.tight_layout()
 #%%
from matplotlib import rc, rcParams

rcParams['axes.labelsize'] = 8
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

#%%
import matplotlib.gridspec as gridspec
import scipy.fftpack as sf
lw=1.5
fs=10
off=1
gs = gridspec.GridSpec(3,8)
plt.figure(figsize=(4.5,1.7*1))
gs.update(left=0.1, right=0.975, bottom=0.12, top=0.95, wspace=1.7, hspace=0.15)  
plt.ylabel('PSD [au]')
El = 1839
tvals = np.linspace(0.1e-6,5000e-6, 900)
Omega = 12*El
delta = 6*El
epsilon = 9*El
pops = tdse.TDSE(tvals,Omega,delta,epsilon)
pops -= np.mean(pops, axis=0)
N = len(pops)
dt = tvals[2]-tvals[1]
xf = np.linspace(0.0, 1/(2 * dt), N  / 2 )
fpops = np.array([np.abs(sf.fft(pops[:, i])**2) for i in range(3)])    
fpops = fpops/np.max(fpops)
fpops = fpops[:, :int(N/2)].T

plt.subplot(gs[0,5:8])
plt.xlim([0,80])
plt.plot(xf*1e-3, fpops[:,0], '-', linewidth=lw, color='darkslateblue')
#plt.yticks(np.linspace(0,1,3), fontsize=fs-off)
plt.yticks([])
plt.xticks([])
plt.subplot(gs[1,5:8])
plt.xlim([0,80])
plt.plot(xf*1e-3, fpops[:,1], '-', linewidth=lw, color='black')
#plt.yticks(np.linspace(0,1,3), fontsize=fs-off)
plt.yticks([])
plt.xticks([])
plt.ylabel('$\mathrm{PSD \ [au]}$', fontsize=fs)
plt.subplot(gs[2,5:8])
plt.xlim([0,80])
plt.plot(xf*1e-3, fpops[:,2], '-', linewidth=lw, color='red')
#plt.yticks(np.linspace(0,1,3), fontsize=fs-off)
plt.yticks([])
plt.xticks(np.linspace(0, 80, 5), fontsize=fs-off)
plt.xlabel('$\mathrm{Frequency \ [MHz]}$', fontsize=fs)


plt.subplot(gs[:,2:5])
tvals = np.linspace(0.1e-6,120e-6, 180)
Omega = 10*El
delta = 6*El
epsilon = 9*El
psi = tdse.TDSE(tvals,Omega,delta,epsilon)
plt.plot(tvals*1e6, psi[:,1], '-', color='k', ms=7, linewidth=lw)
plt.plot(tvals*1e6, psi[:,0],'-', color='r', ms=7, linewidth=lw)
plt.plot(tvals*1e6, psi[:,2], '-', ms=7, linewidth=lw, color='darkslateblue')
plt.axis('Tight')
plt.xticks(np.linspace(0,120,4),fontsize=fs-off)
#plt.yticks([0,0.5,1],fontsize=fs-off)
plt.yticks([])
plt.xlabel('$\mathrm{Pulse \ time} \ [\mu s]$', fontsize=fs)
plt.ylabel('$\mathrm{Probability \ amplitude}$', fontsize=fs)


plt.subplot(gs[:,0:2])
pp=np.linspace(-3.5,3.5,200)
for i in range(0,3):
    plt.plot(pp, eigarray(pp,5,i), 'k-', linewidth=lw)
    plt.vlines(6.0/4, eigarray(0,5,0).min()-0.5,eigarray(pp,5,2).max(), linestyle='--', color='gray')
    plt.axis('Tight')
    plt.ylim([eigarray(0,5,0).min()-0.5,eigarray(pp,5,2).max()])
    plt.xticks(np.linspace(pp.min(), pp.max(), 3))
#    plt.yticks([0,20,40])
    plt.xticks([-3.5,0,3.5],fontsize=fs-off)
#    plt.yticks(np.linspace(int(eigarray(0,5,0).min()),int(eigarray(pp,5,2).max()),4),fontsize=fs-off)
    plt.yticks([])
    plt.xlabel('$\mathrm{Quasimomentum} \ [k_L$]', fontsize=fs)
    plt.ylabel('$\mathrm{Energy} \ [E_L]$', fontsize=fs)
#plt.tight_layout()
#plt.savefig('/Users/banano/Documents/UMD/Research/MolmerSorensen/Paper/Figures/fourier2.pdf')


#%%

import matplotlib.gridspec as gridspec
import scipy.fftpack as sf
lw=1.5
fs=10
off=2
e0 = eigarray(0, Omega/El, delta/El, epsilon/El, 0) * 1.839
e1 = eigarray(0, Omega/El, delta/El, epsilon/El, 1) * 1.839
e2 = eigarray(0, Omega/El, delta/El, epsilon/El, 2) * 1.839

gs = gridspec.GridSpec(3,10)
plt.figure(figsize=(4.5,2*1))
#gs.update(left=0.11, right=0.99, bottom=0.2, top=0.95, wspace=200, hspace=0.45)  
plt.ylabel('PSD [au]')
El = 1839
tvals = np.linspace(0.1e-6,1000e-6, 1000)
Omega = 12*El
delta = 6*El
epsilon = 9*El
pops = tdse.TDSE(tvals,Omega,delta,epsilon)
pops -= np.mean(pops, axis=0)
N = len(pops)
dt = tvals[2]-tvals[1]
xf = np.linspace(0.0, 1/(2 * dt), N  / 2 )
fpops = np.array([np.abs(sf.fft(pops[:, i])**2) for i in range(3)])    
fpops = fpops/np.max(fpops)
fpops = fpops[:, :int(N/2)].T
#%%
plt.subplot(gs[0,6:10])
plt.xlim([0,80])
plt.plot(xf*1e-3, fpops[:,0], '-', linewidth=lw, color='darkslateblue')
#plt.yticks(np.linspace(0,1,3), fontsize=fs-off)
plt.yticks([0],fontsize=fs-off)
plt.ylim([0,1])
plt.xticks([])
plt.subplot(gs[1,6:10])
plt.xlim([0,80])
plt.plot(xf*1e-3, fpops[:,1], '-', linewidth=lw, color='black')
#plt.yticks(np.linspace(0,1,3), fontsize=fs-off)
plt.ylim([0,1])
plt.yticks([0],fontsize=fs-off)
plt.xticks([])
plt.ylabel('$\mathrm{PSD \ [au]}$', fontsize=fs)
plt.subplot(gs[2,6:10])
plt.xlim([0,65])
plt.plot(xf*1e-3, fpops[:,2], '-', linewidth=lw, color='red')
plt.ylim([0,1])
plt.yticks([0],fontsize=fs-off)
plt.xticks([e2-e1, e1-e0, e2-e0], fontsize=fs-off)
plt.xlabel('$\mathrm{Frequency}$', fontsize=fs)


plt.subplot(gs[:,2:6])
tvals = np.linspace(0.1e-6,120e-6, 180)
Omega = 10*El
delta = 6*El
epsilon = 8*El
psi = tdse.TDSE(tvals,Omega,delta,epsilon)
plt.plot(tvals*1e6, psi[:,1], '-', color='k', ms=7, linewidth=lw)
plt.plot(tvals*1e6, psi[:,0],'-', color='r', ms=7, linewidth=lw)
plt.plot(tvals*1e6, psi[:,2], '-', ms=7, linewidth=lw, color='darkslateblue')
plt.axis('Tight')
plt.xticks(np.linspace(0,120,3), 
           ['$t_0$', '$t_1$', '$t_2$'],fontsize=fs)
#plt.yticks([0,0.5,1],fontsize=fs-off)
plt.yticks([0,1], fontsize=fs-off)
plt.xlabel('$\mathrm{Evoltution \ time }$', fontsize=fs)
plt.ylabel('$\mathrm{Probability}$', fontsize=fs)
#plt.text(0.2,1.1, '$\mathrm{b)}$', fontsize=fs)

plt.subplot(gs[:,0:2])

#e0 = eigarray(6,12/np.sqrt(2),0) * 1.839
#e1 = eigarray(6.0/4,12/np.sqrt(2),1) * 1.839
#e2 = eigarray(6.0/4,12/np.sqrt(2),2) * 1.839
plt.hlines(e2 - e0, 0.25 , 0.75, color='darkslategray', linewidth=2)
plt.hlines(e1 - e0, 0.25 , 0.75, color='teal', linewidth=2)
plt.hlines(e0 - e0 ,0.25 , 0.75, color='mediumturquoise', linewidth=2)
plt.xticks([])
plt.yticks([e0-e0,e1-e0,e2-e0], 
           ['$E_1^{\prime}$', '$E_2^{\prime}$', 
           '$E_3^{\prime}$'], fontsize=fs)
plt.ylim([-5,65])
plt.xlim([0,1])
#plt.text(0.2,71.7, '$\mathrm{b)}$', fontsize=fs)

plt.tight_layout()

#plt.xlabel('$\mathrm{Quasimomentum} \ [k_L$]', fontsize=fs)
plt.ylabel('$\mathrm{Energy}$', fontsize=fs)

plt.savefig('/Users/banano/Documents/UMD/Research/MolmerSorensen/Paper/Figures/fig1v3.pdf')

#%%
plt.figure(figsize=(1.2,1.4))
pp=np.linspace(-3.5,3.5,200)
for i in range(0,3):
    plt.plot(pp, eigarray(pp,4,i)-eigarray(0,4,0), 'k-', linewidth=lw)
#    plt.vlines(6.0/4, eigarray(0,5,0).min()-0.5,eigarray(pp,5,2).max(), linestyle='--', color='gray')
    plt.axis('Tight')
    plt.ylim([-5,50])
    plt.xticks(np.linspace(pp.min(), pp.max(), 3))
#    plt.yticks([0,20,40])
    plt.xticks([-3.5,0,3.5],fontsize=fs-off)
#    plt.yticks(np.linspace(int(eigarray(0,5,0).min()),int(eigarray(pp,5,2).max()),4),fontsize=fs-off)
#    plt.yticks([])
    plt.xlabel('$\mathrm{Quasimomentum} \ [k_{\mathrm{L}}$]', fontsize=fs)
    plt.ylabel('$\mathrm{Energy} \ [E_{\mathrm{L}}]$', fontsize=fs)