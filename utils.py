# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:23:49 2017

@author: banano
"""

#from collections import deque
#import sys
#sys.path.append('/Users/banano/Documents/UMD/Research/Rashba/Chern/Utils/')

from scipy.linalg import pinv, lu, solve
import numpy as np
import pandas as pd
import os
#from importlib import reload
from fnmatch import fnmatch
from skimage.feature import blob_log
import matplotlib.pyplot as plt
# from image_reconstruction.cpu_reconstructor import CPUReconstructor as ImageReconstructor
from tqdm import tqdm
import h5py
from skimage.restoration import denoise_tv_chambolle
from numba import jit
from matplotlib.gridspec import GridSpec
import Ramsey_analysis_functions as raf
#reload(raf)
from configparser import ConfigParser
import lmfit
from scipy.ndimage import gaussian_filter
#from ftplib import FTP
#from pathlib import Path

def puck(xyVals, x0, wx, y0, wy) :
    # returns 1.0 within a specified boundary. 0.0 everywhere else:
    #   X Central value                : x0
    #   X width / 2                       : wx
    #   Y Central value                : y0
    #   Y width / 2                        : wy
    condition = (1.0 - ((xyVals[0]-x0)/wx)**2.0 - ((xyVals[1]-y0)/wy)**2.0);
    condition[condition < 0.0] = 0.0;
    condition[condition > 0.0] = 1.0;
    return condition;

def edge(xyVals, x0, wx, y0, wy, thickness) :

    puck1 = puck(xyVals, x0, wx, y0, wy)
    puck2 = puck(xyVals, x0, wx-thickness, y0, wy-thickness)
    edge = (puck1) * (1-puck2)
    return edge

def edge_indices(xyVals, x0, wx, y0, wy, thickness):
    
    indices = []
    edge_vals = edge(xyVals, x0, wx, y0, wy, thickness)
    for i in range(edge_vals.shape[0]):
        for j in range(edge_vals.shape[1]):
            if edge_vals[i,j] > 0:
                indices.append([i,j])

    indices = np.array(indices)
    
    return indices


def reconstruct_probes(mask, raw_probes, raw_atoms):
   reconstructor = ImageReconstructor()
   reconstructor.add_ref_images(raw_probes)
   reconstructed_probes = []
   for atoms in tqdm(raw_atoms):
       reconstructed_probes.append(reconstructor.reconstruct(image=atoms, mask=mask)[0])
   del reconstructor
   return reconstructed_probes

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

def fringeremoval(img_list, ref_list, mask='all', method='svd'):

    nimgs = len(img_list)
    nimgsR = len(ref_list)
    xdim = img_list[0].shape[0]
    ydim = img_list[0].shape[1]

    if mask == 'all':
        bgmask = np.ones([ydim, xdim])
        # around 2% OD reduction with no mask
    else:
        bgmask = mask

    k = (bgmask == 1).flatten(1)

    # needs to be >float32 since float16 doesn't work with linalg

    R = np.dstack(ref_list).reshape((xdim*ydim, nimgsR)).astype(np.float64)
    A = np.dstack(img_list).reshape((xdim*ydim, nimgs)).astype(np.float64)

    # Timings: for 50 ref images lasso is twice as slow
    # lasso 1.00
    # svd 0.54
    # lu 0.54

    optref_list = []

    for j in range(A.shape[1]):

        if method == 'svd':
            b = R[k, :].T.dot(A[k, j])
            Binv = pinv(R[k, :].T.dot(R[k, :]))  # svd through pinv
            c = Binv.dot(b)
            # can also try linalg.svd()

        elif method == 'lu':
            b = R[k, :].T.dot(A[k, j])
            p, L, U = lu(R[k, :].T.dot(R[k, :]))
            c = solve(U, solve(L, p.T.dot(b)))

        elif method == 'lasso':
            lasso = Lasso(alpha=0.01)
            lasso.fit(R, A)
            c = lasso.coef_

        else:
            raise Exception('Invalid method.')

        optref_list.append(np.reshape(R.dot(c), (xdim, ydim)))

    return optref_list

def blob_detect(img, show=False, **kwargs):

    if kwargs is None:
        kwargs = {'min_sigma': 5,
                  'max_sigma': 20,
                  'num_sigma': 15,
                  'threshold': 0.05}

    # image = plt.imread('blobs2.png')
    # image_gray = rgb2gray(image)

    blobs = blob_log(img, **kwargs)
    # Compute radii in the 3rd column.
    try:
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    except IndexError:
        blobs = None

    if show:

        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, interpolation='nearest', vmax=4e4)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
            ax.add_patch(c)

        plt.show()

    return blobs

def bin_image(image, n_pixels):

    try:

        x_shape, y_shape = image.shape
        image_reshape = image.reshape(int(x_shape / n_pixels), n_pixels,
                                      int(y_shape / n_pixels), n_pixels)
        binned_image = image_reshape.mean(axis=1)
        binned_image = binned_image.mean(axis=2)

    except ValueError:
        print('Image is not divisible by that number of pixels')
        binned_image = image

    return binned_image

def bin_image_arr(image_arr, n_rois, n_pixels):
    
    x, y = image_arr.shape[-2:]
    binned_images = np.empty((n_rois, int(x / n_pixels), int(y/ n_pixels)))
#    print(binned_images.shape)
    for i in range(n_rois):
        
        binned_images[i] = bin_image(image_arr[i], n_pixels)
        
    return binned_images

def banana(x):
    print('banana')

    return x

def data_processing(camera, date, sequence, sequence_type, 
                    scanned_parameter='Raman_pulse_time',
                    long_image=False, redo_prepare=True, filtering=True):
    
    """
    Takes a camera, data and sequence number and returns a dataframe with 
    information such as run number, integrated od, scanned variable, microwave
    lock paremeters and an array with 
    """
    
#    sys.path.append('/Users/banano/Documents/UMD/Research/Rashba/Chern/Analysis/')
#    sys.path.append('C:\Users\Ana\Documents\Research\Chern\Analysis')

    
    
    folder = getfolder(date, sequence)
    print('Your current working directory is %s' %os.getcwd())
    
    outfile = '{}_{}_{:04d}.h5'.format(sequence_type, date, sequence)
    
    try:
        with h5py.File('processed_data/' + outfile, 'r') as f:
            f['data']
            
    except KeyError:
        redo_prepare = True
        print('Data not found, preparing again..')
    except IOError:
        redo_prepare = True
        print('Data not found, preparing again..')

    
    if redo_prepare:
        
        print('Looking for files in %s' %folder)
        scanned_parameters = []
        ods = []
        probe_list = []
        atoms_list = []
        run_number = []
        n_runs = []
        sequence_id = []
        sequence_index = []
        status = []
        integrated_od = []
        od1 = []
        od2 = []
        err = []
        Raman_1B_amp = []
        Raman_2C_amp = []
        Raman_3D_amp = []
        
        try:
            i = 0
            j=0
            for r, file in matchfiles(folder):
                i += 1;
            print('Found %s files in folder'%i)
              
            if i> 0:
                
                print('Preparing {} data...'.format(sequence_type)) 
                for r, file in tqdm(matchfiles(folder)):
                    j+=1
                    with h5py.File(os.path.join(r, file), 'r') as h5_file:
                        
                
                        
                        try:
                            uwaves = h5_file['globals/uwaves'].attrs
                            uwaves_lock = uwaves.get('UWAVES_LOCK', 'False')
                            uwaves_lock = eval(uwaves_lock) #convert string to boolean
                            
                        except KeyError:
                            uwaves_lock = False
                            
                        try:
                            
                            img = h5_file['data']['images' + camera]['Raw'][:]
                        
                        except: 
                            img = np.ones([9,484, 644]) * np.nan
                        
                        try:
                            attrs = h5_file['globals'].attrs
                            scanned_parameters.append(attrs[scanned_parameter])  
                            attrs = h5_file.attrs
                            run_number.append(attrs['run number'])
                            n_runs.append(attrs['n_runs'])
                            sequence_id.append(attrs['sequence_id'])
                            sequence_index.append(attrs['sequence_index'])
#                            print(img.shape)
                        
                        except:
                            print(j)
                            scanned_parameters.append(np.nan)  
                            run_number.append(np.nan)
                            n_runs.append(np.nan)
                            sequence_id.append(np.nan)
                            sequence_index.append(np.nan)
                            
                        img = np.float64(img)
                        # print(img.shape)
#                        print(type(uwaves_lock))
#                        plt.imshow(img[1])
#                        plt.show()
                        try:
#                            print(uwaves_lock)
                            if uwaves_lock:
#                                print('uwaves loop')
                                atoms, probe, bg = img[7], img[6], img[8]
        #                        plt.imshow(atoms)
                                
                                uwave_attrs = h5_file['results/uwave_lock'].attrs
                                od1.append(uwave_attrs['od1'])
                                od2.append(uwave_attrs['od2'])
                                err.append(uwave_attrs['err'])                        
                                
                                
                            else:
#                                print('here in the loop')
                       
                                atoms, probe, bg = img[1], img[0], img[2]
                                od1.append(np.nan)
                                od2.append(np.nan)
                                err.append(np.nan)
                   
                            atoms -= bg
                            probe -= bg
                            probe_list.append(probe.T)
                            atoms_list.append(atoms.T)
                            atoms = atoms.clip(1)
                            probe = probe.clip(1)
                            
                            Isat = 3.550505e3
                            od = -np.log(atoms.clip(1)/probe.clip(1))
                            od += (probe - atoms) / Isat
                            od = od.T
#                            plt.imshow(od)
#                            plt.show()
#                            print(j)
                            integrated_od.append(od.sum())

                            if od.sum() < 2500:
                                ods.append(np.nan * np.ones([644, 484]))
                                status.append(str('no atoms'))
                            else:
                                ods.append(od)
                                status.append(str('good'))
                            
                            try:
                                raman_attrs = h5_file['results/raman_monitor'].attrs
                                Raman_1B_amp.append(raman_attrs['Raman_1B_Vamp'])
                                Raman_2C_amp.append(raman_attrs['Raman_2C_Vamp'])
                                Raman_3D_amp.append(raman_attrs['Raman_3D_Vamp'])
                            
                            except:
                                Raman_1B_amp.append(np.nan)
                                Raman_2C_amp.append(np.nan)
                                Raman_3D_amp.append(np.nan)
                        
                        except Exception as e:
                            status.append(str('bad shot'))
#                            print(e)
                            ods.append(np.nan * np.ones([644, 484]))
                            probe_list.append(np.nan * np.ones([644, 484]))
                            atoms_list.append(np.nan * np.ones([644, 484]))
                            integrated_od.append(np.nan)
                            od1.append(np.nan)
                            od2.append(np.nan)
                            err.append(np.nan)
                            Raman_1B_amp.append(np.nan)
                            Raman_2C_amp.append(np.nan)
                            Raman_3D_amp.append(np.nan)
    #                        print(e)

#            print('Here')        
            df = pd.DataFrame()
            df[scanned_parameter] = scanned_parameters
            df['run_number'] = run_number
#            df['n_runs'] = n_runs
#            df['sequence_id'] = sequence_id
#            df['sequence_index'] = sequence_index
            df['status'] = status
#            df['status'] = df['status'].astype('str') 
            df['integrated_od'] = integrated_od
            df['od1'] = od1
            df['od2'] = od2
            df['err'] = err
            df['Raman_1B_amp'] = Raman_1B_amp
            df['Raman_2C_amp'] = Raman_2C_amp
            df['Raman_3D_amp'] = Raman_3D_amp
        
    #        df = df.dropna()
            df = df.sort_values(by=scanned_parameter)
#            print(df[scanned_paramteer].values)
            ods = np.array(ods)
            probe_list = np.array(probe_list)
            atoms_list = np.array(atoms_list)
            sorting_indices = df.index.values
            sorted_od = ods.copy()
#            print(ods.shape)
            sorted_probe = probe_list.copy()
            sorted_atoms = atoms_list.copy()
           

            for i,idx in enumerate(sorting_indices):
                sorted_od[i] = ods[idx]
                sorted_probe[i] = probe_list[idx]
                sorted_atoms[i] = atoms_list[idx]
            
            if filtering:
                # uncoment this for time dependent filtering
#                sorted_od = image_filtering(sorted_od, 
#                                            filter_type='dynamic_fft',
#                                            t=df[scanned_parameter].values)
                sorted_od = image_filtering(sorted_od)
            
            print('Saving data...')
    
            hf = h5py.File('processed_data/' + outfile, 'w')
            hf.attrs['sequence_id'] = sequence_id[0]
            hf.attrs['n_runs'] = n_runs[0]
            hf.attrs['sequence_index'] = sequence_index[0]
            
            g1 = hf.create_group('sorted_images')
            g1.create_dataset('sorted_od',data=sorted_od, compression="gzip")
#            g1.create_dataset('sorted_atoms', data=sorted_atoms, compression='gzip')
            g2 = hf.create_group('data')
            
            
            for key in tqdm(df.keys()):
                try:
                    g2.create_dataset(str(key), data=df[key].values)
                except Exception as e:
#                    print(e)
                    g2.create_dataset(str(key), data=df[key].values.astype('S'))
                    
            hf.close()
#            df.to_hdf('results/' + outfile, 'data', mode='w')
            
        except Exception as e:
#            print('Fix your analysis')
            print(e)        

#    
    else:
        
        print('Loading processed data...')
        df = pd.DataFrame()
        hf = h5py.File('processed_data/' + outfile, mode='r')
        try: 
            g1 = hf.get('sorted_images')
            sorted_od = g1.get('sorted_od')
            sorted_od = np.array(sorted_od)
    #        sorted_atoms = g1.get('sorted_atoms')
    #        sorted_atoms = np.array(sorted_atoms)
            g2 = hf.get('data')
            for key in g2.keys():
                df[key] = np.array(g2.get(key))
            hf.close()
        except:
            print('Could not find data')
            hf.close()

    


    
    if long_image:
        long_image = np.hstack(sorted_od)
        plt.xticks([])
        plt.yticks([])
        plt.imsave(sequence_id[0]+'_long_image.jpg', long_image)
        
    return df, sorted_od#, sorted_atoms#, sorted_probe


def bin_trace(trace_arr, binsize):
    
    binned_trace = []
    i = 0
    try:
        while i < trace_arr.shape[0]:
            val = np.nanmean(trace_arr[i:i+binsize], axis=0)
            binned_trace.append(val)
            i+=binsize
            
    except IndexError:
        
        last_index= len(trace_arr) % binsize
        val = np.nanmean(trace_arr[i:i+last_index], axis=0)
        binned_trace.append(val)
        i+=binsize
        
    return np.array(binned_trace)


def weighted_bin_trace(trace_arr, weight_arr, binsize):
    
    binned_trace = []
#    norm = []
    i = 0
    try:
        while i < trace_arr.shape[0]:
            val = np.nansum((trace_arr*weight_arr)[i:i+binsize], axis=0)
            norm = np.nansum(weight_arr[i:i+binsize], axis=0)
            binned_trace.append(val/norm)
#            norm.append()
            i+=binsize
            
    except IndexError:
        
        last_index= len(trace_arr) % binsize
        val = np.nansum((trace_arr*weight_arr)[i:i+last_index], axis=0)
        norm = np.nansum(weight_arr[i:i+last_index], axis=0)
        binned_trace.append(val/norm)
        i+=binsize
        
    return np.array(binned_trace)


def image_filtering(od_array, f_cutoff=100, filter_type='chambolle', t=0.):
    
    filtered_od_array = od_array.copy()
    print('Filtering images ...')
    for i in tqdm(range(od_array.shape[0])):
        od = od_array[i]
        
        if filter_type == 'chambolle':
             filtered_od_array[i] = denoise_tv_chambolle(od, weight=0.05, 
                                                multichannel=False)
            
             
        elif filter_type == 'fft':
            fft = np.fft.fftshift(np.fft.fft2(od-od.mean()))
            y0, x0 = od.shape
            xy_grid = np.meshgrid(range(x0), range(y0)) 
            x0 = int(x0/2)
            y0 = int(y0/2)
            mask = puck(xy_grid, x0, f_cutoff[i], y0, f_cutoff[i])
#            cutoff = 100
#            mask = np.zeros(od.shape)
#            mask[y0-cutoff:y0+cutoff, x0-cutoff:x0+cutoff]=np.ones([2*cutoff, 2*cutoff])
            fft_filter = fft * mask
#            fft_plot = np.fft.fftshift(np.fft.fft2(od-od.mean())) * mask
            psd = np.abs(fft[y0-f_cutoff:y0+f_cutoff, 
                                         x0-f_cutoff:x0+f_cutoff])**2
#            psd /= psd.max()
            plt.imshow(psd*mask, vmin=0, vmax=0.01)
            plt.show()
            filtered_od_array[i] = np.abs(np.fft.ifft2(fft_filter)+od.mean())
        
        elif filter_type == 'dynamic_fft':
            pad_size = 400
            padded_od = np.zeros([od.shape[0]+2*pad_size, od.shape[1]+2*pad_size])
            padded_od[pad_size:-pad_size,pad_size:-pad_size] = od
            y0, x0 = padded_od.shape
            x0 = int(x0/2)
            y0 = int(y0/2)
            w_mask = 30
            wx = 160
            wy = 160
            zx = np.array([0.33021331, 0.11904938])
            xy = np.array([ 0.01578217, -0.24338255])
            zy = zx + xy
            fft = np.fft.fftshift(np.fft.fft2(padded_od))
            psd = np.abs(fft)**2
            mask = puck(grid(padded_od), x0, w_mask, y0, w_mask)
            mask = gaussian_filter(mask, sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0+zy[1]*t[i]-4, w_mask-3, 
                                           y0+zy[0]*t[i]+3, w_mask-3),sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0-zy[1]*t[i]+4, w_mask-3, 
                                           y0-zy[0]*t[i]-3, w_mask-3),sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0+xy[1]*t[i]-6, w_mask-3, 
                                           y0+xy[0]*t[i]+0.4, w_mask-3),sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0-xy[1]*t[i]+6, w_mask-3, 
                                           y0-xy[0]*t[i]-0.4, w_mask-3),sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0+zx[1]*t[i]+3, w_mask-3, 
                                           y0+zx[0]*t[i]+5, w_mask-3),sigma=2)
            mask += gaussian_filter(puck(grid(padded_od), 
                                           x0-zx[1]*t[i]-3, w_mask-3, 
                                           y0-zx[0]*t[i]-5, w_mask-3),sigma=2)
            mask = np.clip(mask,0,1)
            ifft = (np.fft.ifft2(np.fft.ifftshift(fft*mask)))
#            plt.imshow((mask*psd)[y0-wy:y0+wy,x0-wx:x0+wx], vmax=5e4)
#            plt.show()
#            plt.imshow(ifft.real)
#            plt.colorbar()
#            plt.title('ifft')
#            plt.show()
#            plt.imshow(padded_od)
#            plt.colorbar()
#            plt.title('initial')
#            plt.show()
#            plt.title('diferences')
#            plt.imshow(padded_od-ifft.real)
#            plt.colorbar()
#            plt.show()       
            filtered_od_array[i] = np.abs(ifft)[pad_size:-pad_size,
                                                pad_size:-pad_size]
        
        else:
            print('Filter type not valid')
            
    return filtered_od_array

def rot(kx, ky, theta):
    #TODO precompute 2*pi/360 or use radians
    kx_rot = kx*np.cos((theta))- ky*np.sin((theta))
    ky_rot = kx*np.sin((theta))+ ky*np.cos((theta))
    
    return kx_rot, ky_rot

def shear(kx, ky, theta, shear_param):
    kx_rot, ky_rot = rot(kx, ky, theta) 
    kx_rot_rescaled = kx_rot*np.exp(shear_param)#(1+shear_param)
    kx_inv_rot, ky_inv_rot = rot(kx_rot_rescaled, ky_rot, -theta)

    return  kx_inv_rot, ky_inv_rot  


def shear_inv(kx, ky, theta, shear_param):
    kx_rot, ky_rot = rot(kx, ky, theta) 
    kx_rot_rescaled = kx_rot/(1+shear_param)
    kx_inv_rot, ky_inv_rot = rot(kx_rot_rescaled, ky_rot, -theta)

    return  kx_inv_rot, ky_inv_rot 


#@jit
#def y_val_transform(y_initial, kx_init, kx_trans,  ky_init, ky_trans, 
#                    shear_param, theta, n_cut, sigma_squared):
#
#
#    """
#    y_initial : array that will be transformed, the shape is larger than final transformed
#        array by 2*n_cut in each axis
#    kx_init, ky_init: initial (sheared) x and y (momentum) coordinates of the data that will be transformed
#    kx_trans, ky_trans: (final) transformed x and y (momentum) coordinates of y
#    shear_param: shear parameter that is passed to the shearing function
#    theta: shearing angle
#    n_cut: number of neighbooring array elements used to compute the 
#           transformed array at every coordinate.
#    sigma_squared: squared uncertainty in kx,ky 
#    """    
#    
#    kx_trans_inv, ky_trans_inv = shear(kx_trans, ky_trans, theta, shear_param)
##    kx_trans_inv, ky_trans_inv = shear(kx_init, ky_init, theta, shear_param)
#    
#    gs = GridSpec(1, 3)
#    plt.figure(figsize=(14, 4))
#    plt.subplot(gs[0])
#    k_L = 2*np.pi/790e-9
#    plt.scatter(kx_init.flatten()/k_L, ky_init.flatten()/k_L)
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_init')
#    plt.ylabel('ky_init')
#
#    plt.subplot(gs[1])
#    k_L = 2*np.pi/790e-9
#    plt.scatter(kx_trans.flatten()/k_L, ky_trans.flatten()/k_L)
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_trans')
#    plt.ylabel('ky_trans')
# 
#    plt.subplot(gs[2])
#    k_L = 2*np.pi/790e-9
#    plt.scatter(kx_trans_inv.flatten()/k_L, ky_trans_inv.flatten()/k_L)
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_trans_inv')
#    plt.ylabel('ky_trans_inv')
#    plt.show()
#
#
#    y_prime = np.zeros([kx_trans.shape[0], kx_trans.shape[1]]) #shape of the final array should be the same as the transformed momenta
#    y_shape, x_shape = y_prime.shape
#    
#    for j in (range(y_prime.shape[0])):
#        for i in range(y_prime.shape[1]):
#            y_save = 0.0
#            norm = 0.0
#            
#            kx_init_diff = kx_trans - kx_trans_inv[j,i]
#            ky_init_diff = ky_trans - ky_trans_inv[j,i]
#            diffs = np.sqrt(kx_init_diff**2 + ky_init_diff**2)  #does not assume initial coord. system is uniform
#            min_idx = np.unravel_index(np.argmin(diffs, axis=None), diffs.shape)
##            print(min_idx)
##            min_idx = [j,i]
#            for k in range(max(0, min_idx[0]-n_cut), min(min_idx[0]+n_cut, y_shape)):
#                for l in range(max(0, min_idx[1]-n_cut), min(min_idx[1]+n_cut, x_shape)):
#                     
#                    exp = np.exp(-0.5*(kx_trans_inv[j, i]-kx_init[k,l])**2/sigma_squared)  
#                    exp *=  np.exp(-0.5*(ky_trans_inv[j, i]-ky_init[k,l])**2/sigma_squared) 
#                    y_save +=  exp * y_initial[k,l]
#                    norm +=  exp   
#
#            if norm == 0:
#                y_prime[j, i] = 0.0 #avoid zero division errors. 
#            else:
#                y_prime[j,i] = y_save / norm
#                
#    return y_prime
#
@jit
def y_val_transform(y_initial, kx_init, kx_trans,  ky_init, ky_trans, 
                    shear_param, theta, n_cut, sigma_squared):


    """
    y_initial : array that will be transformed, the shape is larger than final transformed
        array by 2*n_cut in each axis
    kx_init, ky_init: initial (sheared) x and y (momentum) coordinates of the data that will be transformed
    kx_trans, ky_trans: (final) transformed x and y (momentum) coordinates of y
    shear_param: shear parameter that is passed to the shearing function
    theta: shearing angle
    n_cut: number of neighbooring array elements used to compute the 
           transformed array at every coordinate.
    sigma_squared: squared uncertainty in kx,ky 
    """    
    
    kx_trans_inv, ky_trans_inv = shear(kx_trans, ky_trans, theta, shear_param)

#    gs = GridSpec(1, 3)
#    plt.figure(figsize=(14, 4))
#    plt.subplot(gs[0])
#    k_L = 1# 2*np.pi/790e-9
#    plt.scatter(kx_init.flatten(), ky_init.flatten())
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_init')
#    plt.ylabel('ky_init')
#
#    plt.subplot(gs[1])
##    k_L = 2*np.pi/790e-9
#    plt.scatter(kx_trans.flatten(), ky_trans.flatten())
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_trans')
#    plt.ylabel('ky_trans')
# 
#    plt.subplot(gs[2])
##    k_L = 2*np.pi/790e-9
#    plt.scatter(kx_trans_inv.flatten(), ky_trans_inv.flatten())
#    plt.grid(color='k', linestyle='--', linewidth=1)
##    plt.axes().set_aspect('equal')
#    plt.xlabel('kx_trans_inv')
#    plt.ylabel('ky_trans_inv')
#    plt.show()


    y_prime = np.zeros([kx_trans.shape[0], kx_trans.shape[1]]) #shape of the final array should be the same as the transformed momenta
    y_shape, x_shape = y_prime.shape
    
    for j in (range(y_prime.shape[0])):
        for i in range(y_prime.shape[1]):
            y_save = 0.0
            norm = 0.0
            
            kx_init_diff = kx_init - kx_trans_inv[j,i]            
            ky_init_diff = ky_init - ky_trans_inv[j,i]
            diffs = np.sqrt(kx_init_diff**2 + ky_init_diff**2)  #does not assume initial coord. system is uniform
#            plt.imshow(np.abs(diffs))
#            plt.colorbar()
#            plt.show()
            min_idx = np.unravel_index(np.argmin(diffs, axis=None), diffs.shape)
#            min_idx = [j,i]
            for k in range(max(0, min_idx[0]-n_cut), min(min_idx[0]+n_cut, y_shape)):
                for l in range(max(0, min_idx[1]-n_cut), min(min_idx[1]+n_cut, x_shape)):
                     
                    exp = np.exp(-0.5*(kx_trans_inv[j, i]-kx_init[k,l])**2/sigma_squared)  
                    exp *=  np.exp(-0.5*(ky_trans_inv[j, i]-ky_init[k,l])**2/sigma_squared) 
                    y_save +=  exp * y_initial[k,l]
                    norm +=  exp   

            if norm == 0:
                y_prime[j, i] = 0.0 #avoid zero division errors. 
            else:
                y_prime[j,i] = y_save / norm
                
    return y_prime

def shear_array(array, x0, y0, shear, plotme=False):
    
#    params = pars.valuesdict()
#    shear = params['shear']
#    x0 = params['x0']
#    y0 = params['y0']
    n_cut = 5
    theta = (-(90-63.8))*np.pi/180
    
    x_size = array.shape[1]/2
    y_size = array.shape[0]/2


    #
    # Generate Initial grid on which the data was measured
    # 
    indices_x = np.linspace(-x_size, x_size, 2*x_size, endpoint=True)+x0 
    indices_y = np.linspace(-y_size, y_size, 2*y_size, endpoint=True)+y0 #+17
    kx_initial, ky_initial = np.meshgrid(indices_x, indices_y)

    #
    # Generate the final grid on which we want to know the unsheared data
    #
    indices_x = np.linspace(-x_size, x_size, 2*x_size, endpoint=True)+x0
    indices_y = np.linspace(-y_size, y_size, 2*y_size, endpoint=True)+y0
    kx_final, ky_final = np.meshgrid(indices_x, indices_y)

    sigma_squared = np.diff(kx_initial
                            [0])[0]**2 
    new_array = y_val_transform(array, kx_initial, kx_final,  
                    ky_initial, ky_final, 
                    -shear, theta, n_cut, sigma_squared)#+x0
       
    res = (new_array - array)[n_cut:-n_cut, n_cut:-n_cut]
    if plotme:
        plt.figure(figsize=(4.6*3,3))
        gs = GridSpec(1, 3)
        plt.subplot(gs[0])
        plt.title('Initial array')
        old_array = array[n_cut:-n_cut, n_cut:-n_cut]
        plt.imshow(old_array, vmin=old_array.min(), vmax=old_array.max())
        plt.colorbar()  
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        plt.subplot(gs[1])
        plt.title('Final array')
        plt.imshow(new_array[n_cut:-n_cut, n_cut:-n_cut])
        plt.colorbar()  
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        plt.subplot(gs[2])
        plt.title('Difference')
        plt.imshow((res), 
                   vmin=-np.abs(res).max(), vmax=np.abs(res).max(),
                   cmap='RdBu')#
        plt.colorbar() 
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        plt.show()
    
    return new_array
    
def make_rois_array(od_array, x_rois, y_rois, n_rois, wx, wy, 
                    only_one=False, weights=None):#, unshear=False, 
#                    k_arrays=None, shears=[0.07, 0, -0.07]):
#    if unshear == True:
#        print('Undoing shears from SG......')
    if len(x_rois) != len(y_rois):
        print('Wrong roi center vector')
    elif len(x_rois) != n_rois:
        print('Number of rois doesnt match roi center vectors')
    else:
        
        if weights is None:
            weights = np.ones(n_rois)
        
#        weights = np.arry(weights)
        try:
            x_rois = x_rois.astype(int)
            y_rois = y_rois.astype(int)
            
            if only_one:
#   
                rois_array = np.zeros([1, n_rois, 2*wy, 2*wx])
                od = od_array

                for i in range(n_rois):
                    
                    roi = od_array[y_rois[i]-wy:y_rois[i]+wy,
                               x_rois[i]-wx:x_rois[i]+wx]
                    rois_array[0,i,:,:] = roi * weights[i]

            else:
                rois_array = np.zeros([od_array.shape[0], n_rois, 2*wy, 2*wx])
                for j,od in enumerate(od_array):
#                    print(od.shape)
                    for i in range(n_rois):
#                        print(y_rois[i]-wy)
                        roi = od[y_rois[i]-wy:y_rois[i]+wy, x_rois[i]-wx:x_rois[i]+wx]
#                        print(roi.shape)
                        rois_array[j, i,:,:] = roi * weights[i]

            return rois_array

        except Exception as e:
            print(e)
            print('Your roi size is larger than the camera or your roi centers are stupid')
            return np.ones([od_array.shape[0], n_rois, 2*wy, 2*wx])
        
        

def gaussian2D(xy_vals, bkg, amp, x0, sigmax, y0, sigmay) :

    gauss2D = bkg + amp*np.exp(-1*(xy_vals[:,1]-x0)**2/(2*sigmax**2)
                                -1*(xy_vals[:, 0]-y0)**2/(2*sigmay**2))
    
    return gauss2D

def gaussian2D_2(xy_vals, bkg2, amp2, x02, sigmax2, y02, sigmay2) :

    gauss2D = bkg2 + amp2*np.exp(-1*(xy_vals[:,1]-x02)**2/(2*sigmax2**2)
                                -1*(xy_vals[:, 0]-y02)**2/(2*sigmay2**2))
    
    return gauss2D

def gaussian_TF_2D(xy_vals, bkg, amp_g , amp_tf, 
                x0, sigmax, rx_tf, y0, sigmay, ry_tf) :

    gauss2D = bkg + amp_g*np.exp(-1*(xy_vals[:,1]-x0)**2/(2*sigmax**2)
                                -1*(xy_vals[:, 0]-y0)**2/(2*sigmay**2))
    
    condition = (1 - (xy_vals[:,1]-x0) **2 / rx_tf - 
                   (xy_vals[:,0]-y0) **2 / ry_tf) 
    condition[condition <= 0.0] = 0.0
#    
#    plt.plot(condition)
#    plt.show()
    
    TF = amp_tf *(condition)**(3.0/2.0)
    
    return gauss2D + TF

def make_xy_grid(image):
    
    x, y = image.shape
    data = np.empty((x * y, 2))
    x = np.arange(x)
    y = np.arange(y)
    xx, yy = np.meshgrid(x, y)
    data[:,0] = xx.flatten()
    data[:,1] = yy.flatten()
    
    return data

def grid(image):
    
    x, y = image.shape
    x_vec = range(x)
    y_vec = range(y)
    
    return np.array(np.meshgrid(y_vec, x_vec))

def residuals(pars, x, img=None):
    
    params = pars.valuesdict()
    bkg = params['bkg']
    amp = params['amp']
    x0 = params['x0']
    sigmax = params['sigmax']
    y0 = params['y0']
    sigmay = params['sigmay']   
    x1 = params['x1']
    x2 = params['x2']
    x3 = params['x3']
    y1 = params['y1']
    y2 = params['y2']
    y3 = params['y3']
    w = params['w']
    s1 = params['s1']
    s2 = params['s2']
    s3 = params['s3']
    s = np.array([s1, s2, s3])
    x_rois = np.array([x1, x2, x3])
    y_rois = np.array([y1, y2, y3])
    n_rois = 3
    wx = w
    wy = w
    if img is None:
        
        fake_img = np.zeros([644, 484])
        rois = make_rois_array(fake_img, x_rois, y_rois, 
                                 n_rois, wx, wy, only_one=True)[0]#, 
    
    else:
        img = img.reshape((644, 484), order='F')
        rois = make_rois_array(img, x_rois, y_rois, 
                                     n_rois, wx, wy, only_one=True)[0]#, 
#                                 weights=np.array([1, 0.9245, 1.11739]))
#    print(rois.shape)
    rois_sum = np.zeros_like(rois)
    for i in range(3):
        rois_sum[i] = rois[i] * s[i]
    rois_sum = rois_sum.sum(axis=0)
#    plt.imshow(rois_sum)
#    plt.show()
#    rois_sum = rois.sum(axis=1)
    data = rois_sum.ravel(order='F')
    model = gaussian2D(make_xy_grid(rois_sum), bkg, amp, x0, sigmax, y0, sigmay)
    
    if img is None:
        return model

    else:
        return (model - data)

def dict_to_arr(dictionary):
    
    arr = []
    for key in dictionary.keys():
        arr.append(dictionary[key])
        
    return arr

def sequence_preparing(config_file,
                       scanned_parameter, 
                       sequence_type, 
                       data_label,
                       redo_prepare, 
                       redo_h5, 
                       plot_data, 
                       gaussian_reconstruction,
                       plot_rois, 
                       bin_image, 
                       n_bins, 
                       bin_time, 
                       n_bins_t, 
                       undo_shear, 
                       w_fit):


    h5_file_name = 'processed_data/' + sequence_type + '.h5'
    
    if not redo_h5:
        try:
            print('Loading data from h5 file...')
            keys = ['t', 'sorted_od', 'rois_array']
            data_dict = raf.h5_to_dict(h5_file_name, keys)
            t = data_dict['t']
            sorted_od = data_dict['sorted_od']
            rois_array = data_dict['rois_array']
            
        except Exception as e:
            print('Data not found')
            print(e)
            t = np.nan
            sorted_od = np.nan
            rois_array = np.nan
    
    
    else:
        print('Preparing data again')
        parser = ConfigParser()
        parser.read(config_file)
#            sections = parser.sections()
        date = np.array(parser.get(data_label, 'date').split(' , '), 
                                   dtype=int)
        
        sequence_indices = []
        for sequence in parser.get(data_label, 'sequence_indices').split(' , '):
            sequence_indices.append(np.array(sequence.split(' '), dtype=int))
        
        x_rois = np.array(parser.get(data_label, 'x_rois').split(' '), dtype=int)
        y_rois = np.array(parser.get(data_label, 'y_rois').split(' '), dtype=int)
        x_offset = np.int(parser.get(data_label, 'x_offset'))
        y_offset = np.int(parser.get(data_label, 'y_offset'))
        
            
        n_rois = len(x_rois)

        sorted_od = []
        camera = 'XY_Mako'
        for i, date in enumerate(date):
            for sequence in sequence_indices[i]:
          
                df, ods = data_processing(camera, date, sequence, sequence_type, 
                                                        scanned_parameter=scanned_parameter,
                                                        long_image=False, redo_prepare=redo_prepare)
                sorted_od.append(ods)

        sorted_od = np.nanmean(np.array(sorted_od), axis=0)
#        t = df[scanned_parameter].values
#        print(t.shape)
#        idx = np.argmax(np.diff(t))
#        print(idx)
#        print(t[idx])
#        print(t[idx+1])
#        print(t)
        
        if bin_time:
            sorted_od = bin_trace(sorted_od, n_bins_t)
            t = bin_trace(df[scanned_parameter].values, n_bins_t)
        else:
            t = df[scanned_parameter].values
        
        nan_idx = np.isfinite(sorted_od[:,0,0])
        t = t[nan_idx]
        sorted_od = sorted_od[nan_idx]
            

        idx = 0
        od = sorted_od[idx]
        n_rois = len(x_rois)
        weights = np.array([1, 0.9245, 1.11739])
        weights = np.array([1, 1, 1])
        yvals = range(od.shape[0])
        xvals = range(od.shape[1])
        xy_grid = np.meshgrid(xvals, yvals)

        wx = w_fit
        wy = w_fit
        mask = np.zeros(od.shape)
        
        if plot_data:
            od = sorted_od[idx]
            plt.imshow(od, vmin=-0.05, vmax=0.9)
            plt.colorbar(label='OD')
            plt.xlabel('x pixel')
            plt.ylabel('y pixel')
            plt.show()
        for x0, y0 in zip(x_rois, y_rois):
            mask += puck(xy_grid, x0, wx, y0, wy)
        
        if plot_data:
        
            plt.imshow(od*mask, vmin=-0.05, vmax=1)
            plt.show()
            
        rois_array = make_rois_array(sorted_od, x_rois, y_rois, n_rois, wx, wy, 
                                           only_one=False, weights=weights)
        if plot_data:
            plt.imshow(rois_array.sum(axis=1)[idx])
            plt.show()

        x_rois += x_offset
        y_rois += y_offset
        
        if plot_rois:
            
            mask = np.zeros(od.shape)
            for x0, y0 in zip(x_rois, y_rois):
                mask += puck(grid(od), x0, wx, y0, wy)
            #mask = mask.clip(0.9)
            plt.title('Rois')
            plt.imshow(sorted_od[0]*mask, vmin=-0.05)
            plt.show()
            
            plt.imshow(sorted_od[sorted_od.shape[0]-10] * mask, vmin=-0.05)
            plt.show()
        
        if undo_shear:
        
            shear_m = 0.051#0.065
            shear_p = -0.051
            n_cut = 5
            shear_size = w_fit +n_cut
            x0 = x_offset
            y0 = y_offset
            print(x0)
            print(y0)
            shears = [shear_m, 0, shear_p]
            for i in tqdm(range(sorted_od.shape[0])):
                for j in range(0,3):
                    x_i = x_rois[j]-shear_size
                    x_f = x_rois[j]+shear_size
                    y_i = y_rois[j]-shear_size
                    y_f = y_rois[j]+shear_size
#                    plt.imshow(sorted_od[i, y_i:y_f, x_i:x_f])
#                    plt.show()
                    sorted_od[i, y_i:y_f, x_i:x_f] = shear_array(sorted_od[i, y_i:y_f, x_i:x_f],
                                                                 x0, 
                                                                 y0, 
                                                                 shears[j], 
                                                                 plotme=False)


        if gaussian_reconstruction:
            print(x_offset)
            print(y_offset)
            print('initial rois:')
            print(x_rois)
            print(y_rois)
            od = sorted_od[0]
            xy_vals = make_xy_grid(od)
            w = 60
            delta_x = 10
        
            
            if n_rois == 3:
                x1, x2, x3 = x_rois + x_offset
                y1, y2, y3 = y_rois + y_offset
                
            else:
                x1, x2 = x_rois
                y1, y2 = y_rois
                x3,y3 = [80, 80]
                
            rois_before_opt = make_rois_array(od, x_rois, y_rois, 
                                             n_rois, w, w, only_one=True,
                                             weights=weights)[0]
            
            for xx in range(2):
                params = lmfit.Parameters()
                params.add('x1', value= x1, vary=True, min=x1-delta_x, max=x1+delta_x)
                params.add('x2', value= x2, vary=False)#, min=x2-delta_x, max=x2+delta_x)
                params.add('x3', value= x3, vary=True, min=x3-delta_x, max=x3+delta_x)
                params.add('y1', value= y1, vary=True, min=y1-delta_x, max=y1+delta_x)
                params.add('y2', value= y2, vary=False)#, min=y2-delta_x, max=y2+delta_x)
                params.add('y3', value= y3, vary=True, min=y3-delta_x, max=y3+delta_x)
                params.add('w', value= w, vary=False)
                params.add('bkg', value= 0.019, vary=False)#, min=-0.1, max=0.1)
                params.add('amp', value=1.034, vary=True, min=0.8, max=1.4)
                params.add('x0', value=w, vary=True, min=w-10, max=w+10)
                params.add('sigmax', value=52, vary=True, min=40, max=60)
                params.add('y0', value=w, vary=True)#, min=w-10, max=w+10)
                params.add('sigmay', value=56,  vary=True)#, min=40, max=60)
                params.add('s1', value=1, vary=False)#, min=0.5, max=1.5)
                params.add('s2', value=1, vary=True)#, min=0.5, max=1.5)
                params.add('s3', value=1, vary=True)#min=0.5, max=1.5)
                minner = lmfit.Minimizer(residuals, params, fcn_args=(xy_vals, od))
                result = minner.minimize(method='powell')
                
                params_dict = result.params.valuesdict()
                x1 = params_dict['x1']
                x2 = params_dict['x2']
                x3 = params_dict['x3']
                y1 = params_dict['y1']
                y2 = params_dict['y2']
                y3 = params_dict['y3']
                w = params_dict['w']
                wx = w
                wy = w
                n_rois=3
                x_rois = np.array([x1, x2, x3], dtype=int)
                y_rois = np.array([y1, y2, y3], dtype=int)
            
            rois_array = make_rois_array(od, x_rois, y_rois, 
                                             n_rois, wx, wy, only_one=True,
                                             weights=weights)[0]
            
            
            
            rois_sum = np.zeros_like(rois_array)
            for i in range(3):
                rois_sum[i] = rois_array[i]*result.params['s%s'%(i+1)].value
            rois_sum = rois_sum.sum(axis=0)
            
            plt.figure(figsize=(3.5*5,3))
            gs = GridSpec(1,5)
            plt.subplot(gs[0])
            plt.imshow(rois_before_opt.sum(axis=0))
            plt.title('Initial rois recombined')
            plt.subplot(gs[2])
            gauss_fit = residuals(result.params, 
                                        make_xy_grid(od)).reshape((2*w,2*w), 
                                        order='F')
            plt.imshow(gauss_fit)
                                
            plt.title('Gaussian fit')
            plt.subplot(gs[3])
            res = residuals( result.params, 
                                 make_xy_grid(od), od).reshape((2*w,2*w), order='F')
            plt.imshow(res, cmap='RdBu',vmin=-0.4, vmax=0.4)
            plt.title('Optimized residuals')
            
            plt.subplot(gs[1])
            plt.imshow(rois_sum)
            plt.title('Optimal rois recombined')
            plt.subplot(gs[4])
            mask = np.zeros(od.shape)
            for x0, y0 in zip(x_rois, y_rois):
                mask += puck(grid(od), x0, wx+10, y0, wy+10)
            #mask = mask.clip(0.9)
            plt.title('Rois')
            plt.imshow(od*mask, vmin=-0.05, vmax=0.75)
            plt.show()
            print(result.params['x0'].value)
            print(result.params['y0'].value)
            print('optimized rois:')
            print(x_rois)
            print(y_rois)
        
        w = w_fit
        wx = w+20*0
        wy = w    
    
        rois_array_start = make_rois_array(sorted_od, x_rois, y_rois, 
                                         n_rois, wx, wy, only_one=False) 
         
        if bin_image:
        
            rois_array = []
            for rois in rois_array_start:
                rois_array.append(bin_image_arr(rois, n_rois, n_bins))
        
            rois_array = np.array(rois_array)
            x_rois = x_rois / n_bins
            y_rois = y_rois / n_bins
            w = int(w / n_bins)
            wx = int(wx / n_bins)
            wy = int(wy / n_bins)
            
        else:
            rois_array = rois_array_start    
        
        with h5py.File(h5_file_name) as h5f:
            
            print('Saving new processed data')
            
            try:
                  
                del h5f['t']
                del h5f['sorted_od']
                del h5f['rois_array']
                
                raf.h5_save(h5f, 'sorted_od', sorted_od)
                raf.h5_save(h5f, 'rois_array', rois_array)
                raf.h5_save(h5f, 't', t)
#                del h5f['prepare_params']
            
            except:
                
            
                raf.h5_save(h5f, 'sorted_od', sorted_od)
                raf.h5_save(h5f, 'rois_array', rois_array)
                raf.h5_save(h5f, 't', t)
#                raf.h5_save(h5f, 'prepare_params', prepare_params)
                    
    return t, sorted_od, rois_array

