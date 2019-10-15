#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:29:50 2019

@author: banano
"""


import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import h5py
from fnmatch import fnmatch

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
#                        print(len(roi_1))
#                        print(len(scanned_parameters))
                        attrs = h5_file['globals'].attrs
                        scanned_parameters.append(attrs[scanned_parameter])  
                        attrs = h5_file.attrs
  
                    
                    except:
#                        print('Something went wrong...')
#                        print(j)
                        scanned_parameters.append(np.nan)  
                        roi_0.append(np.nan)
                        roi_1.append(np.nan)
                        roi_2.append(np.nan)
                        print(scanned_parameters)
                        int_od.append(np.nan)
                    
        df = pd.DataFrame()
        df[scanned_parameter] = scanned_parameters
#        print(df)
        df['roi_0'] = roi_0
#        print(df)
        df['roi_1'] = roi_1
        df['roi_2'] = roi_2
        df['int_od'] = int_od
        df.sort_values(by=scanned_parameter)
    
    except Exception as e:
        print(e)
#        print(len(roi_0))
#        print(len(scanned_parameters))
           
    return df

# 786 nm
date = 20171010
sequence = 88
df = simple_data_processing(date, sequence)
plt.plot(df['Raman_pulse_time']*1e-6, df['roi_0'], 'o')

#782 nm
date = 20171010
sequence = 81
df = simple_data_processing(date, sequence)
plt.plot(df['Raman_pulse_time']*1e-6, df['roi_0'], 'o')
plt.text(0.01,1e3, 'something')
