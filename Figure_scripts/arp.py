#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:46:20 2019

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
from scipy.linalg import eigvalsh, eigh

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


def arp(pars,delta,data=None):
    
    params = pars.valuesdict()
#    delta_shift = delta 
    exponent = -4*np.log(3)*(delta-params['step_time'])/params['rise_time']
    # make it immune to overflow errors, without sacrificing accuracy:
    if np.iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent,700])
    final = delta[-1]
    initial = delta[0]
    y = (final - initial)/(1 + np.exp(exponent)) + initial
    y += params['off']
    y *= params['amp']
                
    if data is None:
        return y
    
    else:
        return y - data

def H_rf(Omega, delta):
    
    H = [[-delta/2, Omega/2],[Omega/2, delta/2]]
    
    return H

def evals(Omega, delta, i):
    
    evals = eigvalsh(H_rf(Omega, delta))[i]
    return evals

def evecs(Omega, delta):
    
    evecs = eigh(H_rf(Omega, delta))[1]
    p0 = np.abs(evecs[:,0])**2
    
    return p0


delta = np.linspace(-10, 10, 2400)
Omega = 2
energies = []
magnetisation = []
populations = []
params = {'Omega': Omega}
for d in delta:
    params['delta'] = d
    h = H_rf(**params)
    en, ev = eigh(h)
    pops = ev*ev.conj()

    populations.append(pops)
    energies.append(en)

energies = np.array(energies)
#mag = np.array(magnetisation).real
mag = np.linspace(0, 10, 300)
pops = np.array(populations)

for i in range(2):
#    plt.scatter(ks, energies[:, i], c=mag[:, i], s=20)
    plt.scatter(delta, energies[:, i], c=pops[:,i,0], s=0.01, cmap='seismic')
plt.xlabel('Momentum in units of $[k_R]$')
plt.ylabel('Energy [kHz/$h$]')






#plt.plot(delta, evals(Omega, delta, 0))
#plt.plot(delta, evals(Omega, delta, 1))

#%%

fig = plt.figure(figsize=(5.9,2.2))
#    
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0,1])

# uwavelock
date = 20180301
sequence = 70
scanned_parameter = 'ARP_FineBias'
df = simple_data_processing(date, sequence, scanned_parameter).dropna()
delta = df[scanned_parameter].values*1.44e3
p1 = df['roi_1']/(df['roi_0'] + df['roi_1'])




params = lmfit.Parameters()
params.add('step_time', value=0)
params.add('rise_time', value=20.)
params.add('amp', value=1/(28.*2))
params.add('off', value=28)
minner = lmfit.Minimizer(arp, params, 
                         fcn_args=(delta, p1), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min(), delta.max(), 200)
step= arp(result.params, delta_resampled)
plt.plot(delta_resampled, step, 'k')
plt.plot(delta, p1, 'o', mec='k', ms=6, mfc='lightgray')
ax.set_yticks([0, 0.5, 1])
plt.xlim([delta.min(), delta.max()])
plt.xlabel('$\delta$ [kHz]')
plt.ylabel(r'$\vert c_e\vert^2$')

ax = plt.subplot(gs[0,0])
delta_resampled = np.linspace(delta.min(), delta.max(), 200)

Omega = 8
energies = []
magnetisation = []
populations = []
params = {'Omega': Omega}
for d in delta_resampled:
    params['delta'] = d
    h = H_rf(**params)
    en, ev = eigh(h)
    pops = ev*ev.conj()

    populations.append(pops)
    energies.append(en)

energies = np.array(energies)
#mag = np.array(magnetisation).real
mag = np.linspace(0, 10, 300)
pops = np.array(populations)

for i in range(2):
#    plt.scatter(ks, energies[:, i], c=mag[:, i], s=20)
    plt.scatter(delta_resampled, energies[:, i], c=pops[:,i,0], s=0.5, cmap='seismic')
plt.xlabel('$\delta$ [kHz]')
plt.ylabel('Energy [kHz/$h$]')
plt.xlim([delta.min(), delta.max()])
plt.tight_layout()
plt.savefig('arp.pdf')