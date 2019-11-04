#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:50:03 2019

@author: banano
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())
x = np.linspace(-1, 1, 2**8)
y = x
x0 = np.zeros(2**8)
y0 = x
x1 = x
y1 = x0

theta = np.pi*0.15
x2 = np.cos(theta)*x0 - np.sin(theta)*y0
y2 = np.sin(theta)*x0 + np.cos(theta)*y0
x, y = np.meshgrid(x, y)
# Compute z to make the pringle surface.
#z = np.sin(-x*y)
z = x**2 - y**2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax._axis3don = False
ax.view_init(elev=20, azim=57.6+10)
ax.plot_surface(x, y*0, y, antialiased=True, rstride=20, cstride=20, color='Gray', 
                alpha=0.2, edgecolors='w', linewidth=0.8)
#ax.plot_surface(x*0, y, x, antialiased=True, rstride=20, cstride=20, color='Gray', 
#                alpha=0.2, edgecolors='w', linewidth=0.8)
ax.plot_surface(x, y, z, antialiased=True, rstride=20, cstride=20, color='Gray', 
                alpha=0.7, edgecolors='w', linewidth=0.8)
ax.plot(x0, y0, x0**2-y0**2, linewidth=2)
ax.plot(y0, x0, -x0**2+y0**2, linewidth=2)
ax.plot(x2, y2, x2**2-y2**2, linewidth=2)

#plt.savefig('gauss_bonet.pdf', transparent=True)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax._axis3don = False
ax.view_init(elev=20, azim=57.6+10)
#ax.plot_surface(x, y*0, y, antialiased=True, rstride=20, cstride=20, color='Gray', 
#                alpha=0.2, edgecolors='w', linewidth=0.8)
ax.plot_surface(x*0, y, x, antialiased=True, rstride=20, cstride=20, color='Gray', 
                alpha=0.2, edgecolors='w', linewidth=0.8)
ax.plot_surface(x, y, z, antialiased=True, rstride=20, cstride=20, color='Gray', 
                alpha=0.7, edgecolors='w', linewidth=0.8)
ax.plot(x0, y0, x0**2-y0**2, linewidth=2)
#ax.plot(y0, x0, -x0**2+y0**2, linewidth=2)

#plt.savefig('gauss_bonet.pdf', transparent=True)
plt.show()