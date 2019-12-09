#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:15:34 2019

@author: banano
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(2.7*0.95,1.4*0.95*1.25))
#    
gs = GridSpec(1, 1)   
ax = plt.subplot(gs[0])

phi = np.linspace(0,2*np.pi)

plt.plot(phi/np.pi, 1+np.cos(phi))
plt.xlabel('$\phi$ in units of $\pi$')
plt.ylabel('Intensity [arb. u]')
plt.tight_layout()
plt.savefig('mach_zender.pdf')