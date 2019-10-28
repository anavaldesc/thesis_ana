# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 09:45:00 2018

@author: Amilson
"""
import numpy as np

# Fundamentals Physics constants
kB = 1.38065*10**(-23);           # Boltzmann´s cte in J/K
e = 1.602*10**(-19);         # carga do eletron 
epsilon0 = 8.854*10**(-12);  # permissividade do vacuo
mu0 = 4*3.1415*10**(-7);     # permeabilidade do vacuo
grav = 9.8;                 # gravity in m/s**2
c = 2.99*10**8;              # light velocity in m/s
h = 6.626*10**(-34);         # in J.s 
hbar = h/(2*np.pi);            # in J.s
kB = 1.38*10**(-23);         # in J/K 
kBuK = 1.38*10**(-29);       # in J/uK 
uB_JT = 9.274*10**(-24);     # magneton de Bohr  J/T
uB = 9.274*10**(-28);        # magneton de Bohr  J/G
m = 9.109*10**(-31);         # massa do eletron 
ms = 1*10**(-3)
um = 1*10**(-6)

# Rb constants
mRb = 1.44316e-25;          # 87Rb mass in kg
Xsec_um = 2.9064*10**(-13);  # 87Rb abs. X section for sigma pol. light (mm**2)
Xsec_um = 2.9064*0.1;       # 87Rb abs. X section for sigma pol. light (um**2)
asRb = 100*0.53*10**(-10);   # 87Rb scaterring length in m
U0 = 4.0*np.pi*(hbar**2)*asRb/mRb  #effective interaction

#trap parameters
fx = 42.1; wx = 2*np.pi*fx;   
fy = 34.3; wy = 2*np.pi*fy;    
fz = 133.1; wz = 2*np.pi*fz;
wbar = (wx*wy*wz)**(1.0/3.0)
