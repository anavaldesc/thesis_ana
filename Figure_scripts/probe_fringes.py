#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:50:53 2019

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

def simple_data_processing(date, sequence, camera, 
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
    imgs = []
     
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
                        img = h5_file['data']['images' + camera]['Raw'][:]
                        img = np.float64(img)
                        imgs.append(img)
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
                        imgs.append(np.nan)
                    
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
           
    return df, imgs


#%%

#fig = plt.figure(figsize=(5.9,2.2))
##    
#gs = GridSpec(1, 2)   
#ax = plt.subplot(gs[0,0])

# zx pulse
date = 20171028
sequence = 12
camera = 'XY_Mako'
camera = 'XY_Flea3'

scanned_parameter = 'MOT_load_time'
df, img = simple_data_processing(date, sequence, camera, scanned_parameter)

probes = []
for i in range(len(img)):
    try:
        atoms, probe, bg = img[i][0], img[i][1], img[i][2]
        probes.append(probe)
        atoms -= bg
        probe -= bg
        od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe)).T
#        od = -np.log(atoms.clip(1)/probe.clip(1))
#        plt.imshow(img[i][0])

        plt.imshow(od, cmap='seismic')
        plt.show()
    except:
        pass
ods = []
for i in range(len(probes)-1):
    probe=probes[i]
    atoms = probes[i+1]
    od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe)).T
    plt.imshow(od, cmap='seismic', vmax=0.75, vmin=-0.75)
    plt.title(i)
    ods.append(od)
    plt.show()

#%%
bad_od = ods[5]
plt.imshow(bad_od, cmap='seismic', vmax=0.75, vmin=-0.75)
#%%
date = 20171101
sequence = 3
camera = 'XY_Mako'
#camera = 'XY_Flea3'

scanned_parameter = 'MOT_load_time'
df, img = simple_data_processing(date, sequence, camera, scanned_parameter)

for i in range(len(img)):
    try:
        
        atoms, probe, bg = img[i][0], img[i][1], img[i][2]
        atoms -= bg
        probe -= bg
        od = -np.log(((atoms < 1) + atoms) / ((probe < 1) + probe)).T
        plt.show()
        plt.imshow(od, cmap='seismic', vmax=0.75, vmin=-0.75)
        plt.show()
    except:
        print('bad shot')
    
#%%

x,y = (od.T).shape
x = np.arange(0,x)
y = np.arange(0,y)
x, y = np.meshgrid(x,y)


fig = plt.figure(figsize=(4,2))
#    
gs = GridSpec(1, 2)   
ax = plt.subplot(gs[0,1])
plt.pcolormesh(x, y, od, cmap='seismic', vmax=0.75, vmin=-0.75)
ax.set_aspect('equal')
plt.colorbar(label='OD', ticks=[-0.7, 0, 0.7])
plt.text(0,644+30,'$\mathbf{b.}$ Mako camera')
plt.xticks([])
plt.yticks([])



x,y = (bad_od.T).shape
x = np.arange(0,x)
y = np.arange(0,y)
x, y = np.meshgrid(x,y)

ax = plt.subplot(gs[0,0])
plt.pcolormesh(x, y, bad_od, cmap='seismic', vmax=0.75, vmin=-0.75)
plt.xticks([])
plt.yticks([])
plt.text(0,644+30,'$\mathbf{a.}$ Flea3 camera')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('probe_comparison.pdf')