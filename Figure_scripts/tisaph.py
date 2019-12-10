#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:02:42 2019

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt

pump_power = [4.73, 6.3, 7.876, 15.75, 17.342, 18.916, 20.455 ]
pump_power = [4.73, 6.3, 7.876, 9.456, 11.024, 12.61, 14.17]

tisaph_power = [0.323, 0.8832, 1.4395, 4.8228, 5.438, 5.8868, 6.2521]
tisaph_power = [0.323, 0.8832, 1.4395, 2.1425,  2.6797, 3.2465,  3.6876]


fig = plt.figure(figsize=(3.5,2.5))
poly = np.polyfit(pump_power, tisaph_power, 1)
plt.plot(pump_power, np.poly1d(poly)(pump_power), 'k')
plt.plot(pump_power, tisaph_power, 'o', mec='k')
plt.xlabel('Pump power [W]')
plt.ylabel('Ti:Saph power [W]')
plt.tight_layout()
#plt.savefig('tisaph_power.pdf')