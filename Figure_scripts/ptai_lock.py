#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:10:59 2019

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
    od1 = []
    od2 = []
    err = []
     
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

                        rois_od_attrs = h5_file['results/uwave_lock'].attrs
                        od1.append(rois_od_attrs['od1'])
                        od2.append(rois_od_attrs['od2'])
                        err.append(rois_od_attrs['err'])

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
                        od1.append(np.nan)
                        od2.append(np.nan)
                        err.append(np.nan)
#                        print(scanned_parameters)
                        int_od.append(np.nan)
                    
        df = pd.DataFrame()
        df[scanned_parameter] = scanned_parameters
#        print(df)
        df['od1'] = od1
        df['od2'] = od2
        df['err'] = err
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
    y *= params['amp']
    y += params['off']
    
    if data is None:
        return y
    
    else:
        return y - data


#%%

fig = plt.figure(figsize=(5.9,2.2))
#    
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0,0])

# uwavelock
date = 20180315
sequence = 87
scanned_parameter = 'ARP_FineBias'
df1 = simple_data_processing(date, sequence, scanned_parameter)

sequence = 88
scanned_parameter = 'ARP_FineBias'
df2 = simple_data_processing(date, sequence, scanned_parameter)

df = pd.concat([df1, df2], axis=0)
#df = df[1::]
p1 = df['od1']
p2 = df['od2']
#%%

delta = df[scanned_parameter]*1.44e3
om=0.1e1
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=-5)
params.add('amp', value=300)
params.add('off', value=50)
minner = lmfit.Minimizer(rabi_peak, params, 
                         fcn_args=(delta, p1), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min(), delta.max(), 200)
peak = rabi_peak(result.params, delta_resampled)
plt.plot(delta_resampled-delta.mean(), peak, 'k--')
plt.plot(delta-delta.mean(), p1, 'o', mec='k', ms=6, mfc='r', label='$\delta_+$')

#%%
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=-7.5)
params.add('amp', value=300)
params.add('off', value=37)
minner = lmfit.Minimizer(rabi_peak, params, 
                         fcn_args=(delta, p2), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min()-1, delta.max()+1, 200)
peak2 = rabi_peak(result.params, delta_resampled)
plt.plot(delta_resampled-delta.mean(), peak2, 'k--')
plt.plot(delta-delta.mean(), p2, 'o', mec='k', ms=6, mfc='b', label='$\delta_-$')

plt.xlabel('$\delta$ [kHz]')
plt.ylabel(r'$n$ [arb. u.]')
plt.xlim([delta.min()-delta.mean(), delta.max()-delta.mean()])
plt.legend()
plt.text(-8.03368421, 173, '$\mathbf{a.}$')
#plt.xlim([-30, 30])
#plt.ylim([0,1])
#ax.set_yticks([0, 0.5, 1])
#plt.text(-30, 1.05, '$\mathbf{a.}$')
#%%
ax = plt.subplot(gs[0,1])
# error signal


plt.plot(delta_resampled-delta.mean(), (peak-peak2)/(peak+peak2), '--', color='k')
plt.plot(delta-delta.mean(), df['err'], 'o', mec='k', ms=6, mfc='lightgray')
plt.xlim([delta.min()-delta.mean(), delta.max()-delta.mean()])
plt.xlabel('$\delta$ [kHz]')
plt.ylabel('$n_{\mathrm{imb}}$')
plt.tight_layout()
ax.set_yticks([-0.5, 0, 0.5])
plt.text(-8.03368421, 0.65, '$\mathbf{b.}$')
plt.savefig('uwave_lock.pdf')
#ax.set_yticklabels([])
#plt.ylim([0,1])
#plt.xlabel('Pulse time [$\mu$s]')
#plt.text(0, 1.05, '$\mathbf{b.}$')
#plt.tight_layout()
#plt.savefig('rabi_cycle.pdf')