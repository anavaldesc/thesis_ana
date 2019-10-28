#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:33:53 2019

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import cts
from scipy.integrate import odeint

def n_tf(x, r_tf, n0):
    
    n = n0*(1 - x**2/r_tf**2)
    n = np.clip(n, 0, np.abs(n))
    return n
x = np.linspace(-2, 2, 500)

fig = plt.figure(figsize=(5.9,2.5))
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0])
plt.plot(x, n_tf(x, 1, 1), linewidth=1)
ax.set_yticks([0, 0.5, 1])
ax.set_xlim([-2, 2])
#ax.set_xticks([-4,-2, 0, 2, 4])
#ax.set
ax.set_xlabel('$x/R_x$')
ax.set_ylabel('$n(x)/n_0$')


""" Def func castin and evaluate: 1st in situ radii, 2nd chem pot, 3rd N atoms"""
def castin(w,t,p):
	L1,y1,L2,y2,L3,y3 = w  #L == lambda, is the expansion coeficient
	wx,wy,wz  = p
	
	f = [y1,((wx*wx)/((L1*L1)*L2*L3)),
		 y2,((wy*wy)/((L2*L2)*L1*L3)),
		  y3,((wz*wz)/((L3*L3)*L1*L2))]
	return f

# Initial conditions for coupled equations
## L is the coeficient for expansion rate R(TOF) = L*R(insitu)
## then at t=0 L=1. y=dL/dt in oder to transform the system of 2nd order diff. eqs.
## a system of 1st order diff. eqs.
w0 = [1.0,0.0,1.0,0.0,1.0,0.0]
p = [cts.wx,cts.wy,cts.wz]

L1,y1,L2,y2,L3,y3 =[[],[],[],[],[],[]]

# ODE solver parameters
tof = 5.0e-3
abserr = 1.0e-9
relerr = 1.0e-6
stoptime = tof
numpoints = 100

# Create the time samples for the output of the ODE solver.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

# Call the ODE solver.
wsol = odeint(castin, w0, t,args=(p,),
              atol=abserr, rtol=relerr)

for t1, w1 in zip(t, wsol):
        L1.append(w1[0])
        y1.append(w1[1])
        L2.append(w1[2])
        y2.append(w1[3])
        L3.append(w1[4])
        y3.append(w1[5])

L1,y1,L2,y2,L3,y3 =[np.hstack(L1),np.hstack(y1),np.hstack(L2),np.hstack(y2),
                    np.hstack(L3),np.hstack(y3)]

t = np.array(t)*1e3
ax = plt.subplot(gs[1])
plt.plot(t, L1, label='$R_x$', linewidth=1)
plt.plot(t, L2, label='$R_y$', linewidth=1)
plt.plot(t, L3, label='$R_z$', linewidth=1)
ax.set_ylabel('$R_i(t)/R_i(t=0)$')
ax.set_xlabel('$t$ [ms]')
ax.set_xlim([0,5])
plt.legend()
plt.tight_layout()
plt.savefig('Thomas_fermi.pdf')

# TF radius (take the last value)
#Rx_0 = rx_cal/(L1[len(L1)-1]);
#Ry_0 = ry_cal/(L2[len(L2)-1]);
#
#print "Rx_0 is: %s" %(Rx_0)
#print "Ry_0 is: %s" %(Ry_0)

## Get the chemical potential
#u_chemx = ((Rx_0**2.0)*cts.mRb*(cts.wx**2.0))/2.0
#u_chemy = ((Ry_0**2.0)*cts.mRb*(cts.wy**2.0))/2.0
#u_chem = (u_chemx+u_chemy)/2

## Get the number of atoms
#N = ((8.0*u_chem*np.pi)/(15.0*cts.U0))*(2*u_chem/(cts.mRb*cts.wbar**2.0))**(3.0/2.0)
#roundN=round(N*1.0/1e6,3)
#
#print "Chemical potential for x is: %s" %(u_chemx)
#print "Chemical potential for y is: %s" %(u_chemy)
#print "Nbec %se6." %(roundN)
#
#
#show_plots = 1
#if show_plots:
#    fig = plt.figure()
#    ax = fig.add_subplot(221)
#    ax.imshow(roi.reshape(roi.shape[0], roi.shape[1]),
#                       vmin=fullOD_low,vmax=fullOD_high,
#                       extent=(figureaxis))
#    ellipse = Ellipse((np.mean(peaksposx)+xlow,np.mean(peaksposy)+ylow),30,30, 
#                  fill = 0, color = "r", linestyle = "dashed",
#                  linewidth = 2)
#    ax.add_patch(ellipse)
#    
#    ax = fig.add_subplot(222)
#    im_profx = ax.plot(xROI, x1D_distribution,'r--', label='roi x profile')
#    fitt_profx = ax.plot(xROI, gaussResults1D_x,'b--', label='fitt_Gauss x profile')
#    im_profx_wo_gauss = ax.plot(xROI, x1D_distribution_wo_gauss,'g--', label='roi w/o gauss x')
#    ax.legend(loc='upper right', shadow=True)
#   
#        
#    ax = fig.add_subplot(223)
#    ax.plot(xROI, x1D_distribution_wo_gauss,'g--', label='roi w/o gauss x')
#    ax.plot(xROI, TFResults1D_x,'b--', label='fitt_TF x profile')
#    ax.legend(loc='upper right', shadow=True)
#            
#
#    ax = fig.add_subplot(224, projection='3d')
#    ax.contour3D(x, y, data_fitted_tf, 20, cmap='binary')
#    ax.contour3D(x, y, roi, 20)
#    fig.show()
