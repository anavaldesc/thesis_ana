#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:32:17 2019

@author: banano
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit

def matchfiles(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if fnmatch(file, '*.h5'):
                yield root, file
        break  # break after listing top level dir


def getfolder(date, sequence):
    date = pd.to_datetime(str(date))
    folder = 'data/' + date.strftime('%Y/%m/%d') + '/{:04d}'.format(sequence)
    return folder

def simple_data_processing(date, sequence, 
                           scanned_parameter='Raman_pulse_time'):
    

    """
    Takes a camera, data and sequence number and returns a dataframe with 
    information such as run number, integrated od, scanned variable, microwave
    lock paremeters and an array with 
    """

    folder = getfolder(date, sequence)
    print('Your current working directory is %s' %os.getcwd())
    
        
    print('Looking for files in %s' %folder)
    scanned_parameters = []
    int_od = []
    roi_0 = []
    roi_1 = []
    roi_2 = []
     
    try:
        i = 0
        j=0
        for r, file in matchfiles(folder):
            i += 1;
        print('Found %s files in folder'%i)
          
        if i> 0:
            
            print('Preparing data...') 
            for r, file in tqdm(matchfiles(folder)):
                j+=1
                with h5py.File(os.path.join(r, file), 'r') as h5_file:
                                        
                    try:
#                        print(j)

                        rois_od_attrs = h5_file['results/rois_od'].attrs
                        roi_0.append(rois_od_attrs['roi_0'])
                        roi_1.append(rois_od_attrs['roi_1'])
                        roi_2.append(rois_od_attrs['roi_2'])
                        int_od.append(rois_od_attrs['opt_depth'])
#                        print(rois_od_attrs['roi_0'])
#                        print(len(roi_1))
#                        print(len(scanned_parameters))
                        attrs = h5_file['globals'].attrs
                        if scanned_parameter == 'delta_xyz_freqs':
                            scanned_parameters.append(attrs[scanned_parameter][0])
                        else:
                            scanned_parameters.append(attrs[scanned_parameter])

                    
                    except:
#                        print('Something went wrong...')
#                        print(j)
                        scanned_parameters.append(np.nan)  
                        roi_0.append(np.nan)
                        roi_1.append(np.nan)
                        roi_2.append(np.nan)
#                        print(scanned_parameters)
                        int_od.append(np.nan)
                    
        df = pd.DataFrame()
        df[scanned_parameter] = scanned_parameters
#        print(df)
        df['roi_0'] = roi_0
#        print(df)
        df['roi_1'] = roi_1
        df['roi_2'] = roi_2
        df['int_od'] = int_od
        df = df.sort_values(by=scanned_parameter)
    
    except Exception as e:
        print(e)
#        print(len(roi_0))
#        print(len(scanned_parameters))
           
    return df

def rabi_peak(pars,delta,data=None):
    
    params = pars.valuesdict()
    delta_shift = delta - params['delta0']
    y = np.sin(np.sqrt(params['om']**2 + delta_shift**2)/2 * params['t'])**2
    y *= params['om']**2 / (params['om']**2 + delta_shift**2) 
    
    if data is None:
        return y
    
    else:
        return y - data

def rabi_flop(pars, t,data=None):
    
    params = pars.valuesdict()
    delta_shift = params['delta0']
    y = np.sin(np.sqrt(params['om']**2 + delta_shift**2)/2 * t)**2
    y *= params['om']**2 / (params['om']**2 + delta_shift**2) 
    y *= np.exp(-t/params['decay'])
    
    if data is None:
        return y
    
    else:
        return y - data

#%%

fig = plt.figure(figsize=(5.9,2.2))
#    
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0,0])

# zx pulse
date = 20171205
sequence = 34
scanned_parameter = 'delta_xyz_freqs'
df = simple_data_processing(date, sequence, scanned_parameter)
df = df[1::]
p1 = df['roi_1']/(df['roi_0'] + df['roi_1'])
#%%

delta = df[scanned_parameter]
om=0.6e1
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=145)
minner = lmfit.Minimizer(rabi_peak, params, 
                         fcn_args=(delta, p1), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min()-10, delta.max()+10, 200)
peak = rabi_peak(result.params, delta_resampled)
plt.plot(delta_resampled-result.params['delta0'], peak, 'k')
plt.plot(df[scanned_parameter]-result.params['delta0'], p1, 'o', mec='k', ms=6)
plt.xlabel('$\delta$ [kHz]')
plt.ylabel(r'$\vert c_e\vert ^2$')
plt.xlim([-30, 30])
plt.ylim([0,1])
ax.set_yticks([0, 0.5, 1])
plt.text(-30, 1.05, '$\mathbf{a.}$')
#%%
ax = plt.subplot(gs[0,1])
# rabi flop
date = 20171206
sequence = 111
scanned_parameter = 'pulse_time'
df = simple_data_processing(date, sequence, scanned_parameter)
p1 = df['roi_1']/(df['roi_0'] + df['roi_1'])
plt.xlim([0, df[scanned_parameter].max()])
params = lmfit.Parameters()
om = 7.17*1e-2*1.23
params.add('om', value=om)
params.add('delta0', value=0.0)
params.add('decay', value=1.2e3)
minner = lmfit.Minimizer(rabi_flop, params, 
                         fcn_args=(delta, p1), nan_policy='omit')
                    
result = minner.minimize('leastsq')
t_resampled = np.linspace(0, 120, 100)
rabi_osc = rabi_flop(result.params, t_resampled)
plt.plot(t_resampled, rabi_osc, 'k')
plt.plot(df[scanned_parameter], p1, 'o', mec='k', ms=6, mfc='#ff7f0e')
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([])
plt.ylim([0,1])
plt.xlabel('Pulse time [$\mu$s]')
plt.text(0, 1.05, '$\mathbf{b.}$')
plt.tight_layout()
plt.savefig('rabi_cycle.pdf')