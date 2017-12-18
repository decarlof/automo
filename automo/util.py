#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #########################################################################
# Copyright (c) 2016, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2016. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module to create basic tomography data analyis automation.

"""

import os, glob
import time
import string
import unicodedata
from distutils.dir_util import mkpath
import re
import logging
import pyfftw
from tomopy import downsample
import tomopy.util.dtype as dtype
import scipy.ndimage as ndimage
import dxchange
import h5py
import six.moves
import warnings
import inspect
import gc
try:
    import netCDF4 as cdf
except:
    pass
import numpy as np
import tomopy.misc.corr
try:
    import xlearn
    from xlearn.classify import model
except:
    warnings.warn('Cannot import package xlearn.')

# logger = logging.getLogger(__name__)
PI = 3.1415927

__author__ = ['Francesco De Carlo', 'Ming Du']
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['append',
           'clean_entry',
           'clean_folder_name', 
           'dataset_info',
           'try_folder',
           'h5group_dims',
           'touch',
           'search_in_folder_dnn']


def h5group_dims(fname, dataset='exchange/data'):
    """
    Read data from hdf5 file array dims for a specific group.

    Parameters
    ----------
    fname : str
        String defining the path of file or file name.
    dataset : str
        Path to the dataset inside hdf5 file where data is located.

    Returns
    -------
    dims : list
        
    """
    try:
        with h5py.File(fname, "r") as f:
            try:
                data = f[dataset]
            except KeyError:
                return None
            shape = data.shape
    except KeyError:
        shape = None
    return shape


def dataset_info(fname):
    """
    Determine the tomographic data set array dimentions    

    Parameters
    ----------
    fname : str
        h5 full path file name.


    Returns
    -------
    dims : list
        List containing the data set array dimentions.
    """
    
    exchange_base = "exchange"
    tomo_grp = '/'.join([exchange_base, 'data'])
    flat_grp = '/'.join([exchange_base, 'data_white'])
    dark_grp = '/'.join([exchange_base, 'data_dark'])
    theta_grp = '/'.join([exchange_base, 'theta'])
    theta_flat_grp = '/'.join([exchange_base, 'theta_white'])
    tomo_list = []
    try: 
        tomo = h5group_dims(fname, tomo_grp)
        flat = h5group_dims(fname, flat_grp)
        dark = h5group_dims(fname, dark_grp)
        theta = h5group_dims(fname, theta_grp)
        theta_flat = h5group_dims(fname, theta_flat_grp)
        tomo_list.append('tomo')
        tomo_list.append(tomo)
        tomo_list.append('flat')
        tomo_list.append(flat)
        tomo_list.append('dark')
        tomo_list.append(dark)
        tomo_list.append('theta')
        tomo_list.append(theta)
        tomo_list.append('theta_flat')
        tomo_list.append(theta_flat)
        return tomo_list
    except OSError:
        pass


def clean_entry(entry):
    """
    Remove from user last name characters that are not compatible folder names.
     
    Parameters
    ----------
    entry : str
        user last name    
    Returns
    -------
    entry : str
        user last name compatible with directory name   
    """

    
    valid_folder_entry_chars = "-_%s%s" % (string.ascii_letters, string.digits)

    cleaned_folder_name = unicodedata.normalize('NFKD', entry.decode('utf-8', 'ignore')).encode('ASCII', 'ignore')
    return ''.join(c for c in cleaned_folder_name if c in valid_folder_entry_chars)

def clean_folder_name(directory):
    """
    Clean the folder name from unsupported characters before
    creating it.
    

    Parameters
    ----------
    folder : str
        Folder that will be containing multiple h5 files.

    """

    valid_folder_name_chars = "-_"+ os.sep + "%s%s" % (string.ascii_letters, string.digits)
    cleaned_folder_name = unicodedata.normalize('NFKD', directory.decode('utf-8', 'ignore')).encode('ASCII', 'ignore')
    
    return ''.join(c for c in cleaned_folder_name if c in valid_folder_name_chars)

def try_folder(directory):
    """
    Function description.

    Parameters
    ----------
    parameter_01 : type
        Description.

    parameter_02 : type
        Description.

    parameter_03 : type
        Description.

    Returns
    -------
    return_01
        Description.
    """

    print ("2")
    try:
        if os.path.isdir(directory):
            return True
        else:
            print(directory + " does not exist")
            a = six.moves.input('Would you like to create ' + directory + ' ? ').lower()
            if a.startswith('y'): 
                mkpath(directory)
                print("Great!")
                return True
            else:
                print ("Sorry for asking...")
                return False
    except: 
        pass # or raise
    else: 
        return False


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def append(fname, process):
    with open(fname, "a") as pfile:
        pfile.write(process)


def entropy(img, range=(-0.002, 0.003), mask_ratio=0.9, window=None, ring_removal=True, center_x=None, center_y=None):

    temp = np.copy(img)
    temp = np.squeeze(temp)
    if window is not None:
        window = np.array(window, dtype='int')
        if window.ndim == 2:
            temp = temp[window[0][0]:window[1][0], window[0][1]:window[1][1]]
        elif window.ndim == 1:
            mid_y, mid_x = (np.array(temp.shape) / 2).astype(int)
            temp = temp[mid_y-window[0]:mid_y+window[0], mid_x-window[1]:mid_x+window[1]]
        # dxchange.write_tiff(temp, 'tmp/data', dtype='float32', overwrite=False)
    if ring_removal:
        temp = np.squeeze(tomopy.remove_ring(temp[np.newaxis, :, :], center_x=center_x, center_y=center_y))
    if mask_ratio is not None:
        mask = tomopy.misc.corr._get_mask(temp.shape[0], temp.shape[1], mask_ratio)
        temp = temp[mask]
    temp = temp.flatten()
    # temp[np.isnan(temp)] = 0
    temp[np.invert(np.isfinite(temp))] = 0
    hist, e = np.histogram(temp, bins=10000, range=range)
    hist = hist.astype('float32') / temp.size + 1e-12
    val = -np.dot(hist, np.log2(hist))
    return val


def minimum_entropy(folder, pattern='*.tiff', range=None, mask_ratio=0.9, window=None, ring_removal=True,
                    center_x=None, center_y=None, reliability_screening=False, verbose=False):

    flist = glob.glob(os.path.join(folder, pattern))
    flist.sort()
    a = []
    s = []
    if range is None:
        temp = dxchange.read_tiff(flist[int(len(flist) / 2)])
        temp = temp.copy()
        temp_std = np.std(temp)
        temp_mean = np.mean(temp)
        temp[np.where(temp > (temp_mean + temp_std * 10))] = temp_mean
        temp[np.where(temp < (temp_mean - temp_std * 10))] = temp_mean
        hist_min = temp.min()
        hist_min = hist_min * 2 if hist_min < 0 else hist_min * 0.5
        hist_max = temp.max()
        hist_max = hist_max * 2 if hist_max > 0 else hist_min * 0.5
        range = (hist_min, hist_max)
        print('Auto-determined histogram range is ({}, {}).'.format(hist_min, hist_max))
    for fname in flist:
        if verbose:
            print(fname)
        img = dxchange.read_tiff(fname)
        # if max(img.shape) > 1000:
        #     img = scipy.misc.imresize(img, 1000. / max(img.shape), mode='F')
        # if ring_removal:
        #     img = np.squeeze(tomopy.remove_ring(img[np.newaxis, :, :]))
        s.append(entropy(img, range=range, mask_ratio=mask_ratio, window=window, ring_removal=ring_removal,
                         center_x=center_x, center_y=center_y))
        a.append(fname)
        gc.collect()
    if reliability_screening:
        if a[np.argmin(s)] in [flist[0], flist[-1]]:
            return None
        elif abs(np.min(s) - np.mean(s)) < 0.2 * np.std(s):
            return None
        else:
            return float(os.path.splitext(os.path.basename(a[np.argmin(s)]))[0])
    else:
        return float(os.path.splitext(os.path.basename(a[np.argmin(s)]))[0])


def read_data_adaptive(fname, proj=None, sino=None, data_format='aps_32id', shape_only=False, return_theta=True, **kwargs):
    """
    Adaptive data reading function that works with dxchange both below and beyond version 0.0.11.
    """
    theta = None
    dxver = dxchange.__version__
    m = re.search(r'(\d+)\.(\d+)\.(\d+)', dxver)
    ver = m.group(1, 2, 3)
    ver = map(int, ver)
    if proj is not None:
        proj_step = 1 if len(proj) == 2 else proj[2]
    if sino is not None:
        sino_step = 1 if len(sino) == 2 else sino[2]
    if data_format == 'aps_32id':
        if shape_only:
            f = h5py.File(fname)
            d = f['exchange/data']
            return d.shape
        try:
            if ver[0] > 0 or ver[1] > 1 or ver[2] > 1:
                dat, flt, drk, theta = dxchange.read_aps_32id(fname, proj=proj, sino=sino)
            else:
                dat, flt, drk = dxchange.read_aps_32id(fname, proj=proj, sino=sino)
                f = h5py.File(fname)
                theta = f['exchange/theta'].value
                theta = theta / 180 * np.pi
        except:
            f = h5py.File(fname)
            d = f['exchange/data']
            theta = f['exchange/theta'].value
            theta = theta / 180 * np.pi
            if proj is None:
                dat = d[:, sino[0]:sino[1]:sino_step, :]
                flt = f['exchange/data_white'][:, sino[0]:sino[1]:sino_step, :]
                try:
                    drk = f['exchange/data_dark'][:, sino[0]:sino[1]:sino_step, :]
                except:
                    print('WARNING: Failed to read dark field. Using zero array instead.')
                    drk = np.zeros([flt.shape[0], 1, flt.shape[2]])
            elif sino is None:
                dat = d[proj[0]:proj[1]:proj_step, :, :]
                flt = f['exchange/data_white'].value
                try:
                    drk = f['exchange/data_dark'].value
                except:
                    print('WARNING: Failed to read dark field. Using zero array instead.')
                    drk = np.zeros([1, flt.shape[1], flt.shape[2]])
            else:
                dat = None
                flt = None
                drk = None
                print('ERROR: Sino and Proj cannot be specifed simultaneously. ')
    elif data_format == 'aps_13bm':
        f = cdf.Dataset(fname)
        if shape_only:
            return f['array_data'].shape
        if sino is None:
            dat = f['array_data'][proj[0]:proj[1]:proj_step, :, :].astype('uint16')
            basename = os.path.splitext(fname)[0]
            flt1 = cdf.Dataset(basename + '_flat1.nc')['array_data'][...]
            flt2 = cdf.Dataset(basename + '_flat2.nc')['array_data'][...]
            flt = np.vstack([flt1, flt2]).astype('uint16')
            drk = np.zeros([1, flt.shape[1], flt.shape[2]]).astype('uint16')
            drk[...] = 64
        elif proj is None:
            dat = f['array_data'][:, sino[0]:sino[1]:sino_step, :].astype('uint16')
            basename = os.path.splitext(fname)[0]
            flt1 = cdf.Dataset(basename + '_flat1.nc')['array_data'][:, sino[0]:sino[1]:sino_step, :]
            flt2 = cdf.Dataset(basename + '_flat2.nc')['array_data'][:, sino[0]:sino[1]:sino_step, :]
            flt = np.vstack([flt1, flt2]).astype('uint16')
            drk = np.zeros([1, flt.shape[1], flt.shape[2]]).astype('uint16')
            drk[...] = 64

    if return_theta:
        return dat, flt, drk, theta
    else:
        return dat, flt, drk


def most_neighbor_clustering(data, radius):

    data = np.array(data)
    counter = np.zeros(len(data))
    for ind, i in enumerate(data):
        for j in data:
            if j != i and abs(j - i) < radius:
                counter[ind] += 1
    return data[np.where(counter == counter.max())]


def find_center_vo(tomo, ind=None, smin=-50, smax=50, srad=6, step=0.5,
                   ratio=0.5, drop=20):
    """
    Transplanted from TomoPy with minor fixes.
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.
    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of the sinogram.
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.
    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)

    if ind is None:
        ind = tomo.shape[1] // 2
    _tomo = tomo[:, ind, :]

    # Enable cache for FFTW.
    pyfftw.interfaces.cache.enable()

    # Reduce noise by smooth filters. Use different filters for coarse and fine search
    _tomo_cs = ndimage.filters.gaussian_filter(_tomo, (3, 1))
    _tomo_fs = ndimage.filters.median_filter(_tomo, (2, 2))

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        _tomo_coarse = downsample(np.expand_dims(_tomo_cs,1), level=2)[:, 0, :]
        init_cen = _search_coarse(_tomo_coarse, smin/4, smax/4, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen*4, ratio, drop)
    else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)

    # logger.debug('Rotation center search finished: %i', fine_cen)
    return fine_cen


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (Nrow, Ncol) = sino.shape
    print(Nrow, Ncol)
    centerfliplr = (Ncol - 1.0) / 2.0

    # Copy the sinogram and flip left right, the purpose is to
    # make a full [0;2Pi] sinogram
    _copy_sino = np.fliplr(sino[1:])

    # This image is used for compensating the shift of sinogram 2
    temp_img = np.zeros((Nrow - 1, Ncol), dtype='float32')
    temp_img[:] = np.flipud(sino)[1:]

    # Start coarse search in which the shift step is 1
    listshift = np.arange(smin, smax + 1)
    print('listshift', listshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    mask = _create_mask(2 * Nrow - 1, Ncol, 0.5 * ratio * Ncol, drop)
    for i in listshift:
        _sino = np.roll(_copy_sino, int(i), axis=1)
        if i >= 0:
            _sino[:, 0:i] = temp_img[:, 0:i]
        else:
            _sino[:, i:] = temp_img[:, i:]
        listmetric[i - smin] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                np.vstack((sino, _sino))))) * mask)
    minpos = np.argmin(listmetric)
    print('coarse return', centerfliplr + listshift[minpos] / 2.0)
    return centerfliplr + listshift[minpos] / 2.0


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    Nrow, Ncol = sino.shape
    centerfliplr = (Ncol + 1.0) / 2.0 - 1.0
    # Use to shift the sinogram 2 to the raw CoR.
    shiftsino = np.int16(2 * (init_cen - centerfliplr))
    _copy_sino = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
    if init_cen <= centerfliplr:
        lefttake = np.int16(np.ceil(srad + 1))
        righttake = np.int16(np.floor(2 * init_cen - srad - 1))
    else:
        lefttake = np.int16(np.ceil(
            init_cen - (Ncol - 1 - init_cen) + srad + 1))
        righttake = np.int16(np.floor(Ncol - 1 - srad - 1))
    Ncol1 = righttake - lefttake + 1
    mask = _create_mask(2 * Nrow - 1, Ncol1, 0.5 * ratio * Ncol, drop)
    numshift = np.int16((2 * srad) / step) + 1
    listshift = np.linspace(-srad, srad, num=numshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    factor1 = np.mean(sino[-1, lefttake:righttake])
    factor2 = np.mean(_copy_sino[0,lefttake:righttake])
    _copy_sino = _copy_sino * factor1 / factor2
    num1 = 0
    for i in listshift:
        _sino = ndimage.interpolation.shift(
            _copy_sino, (0, i), prefilter=False)
        sinojoin = np.vstack((sino, _sino))
        listmetric[num1] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                sinojoin[:, lefttake:righttake + 1]))) * mask)
        num1 = num1 + 1
    minpos = np.argmin(listmetric)
    return init_cen + listshift[minpos] / 2.0


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * PI)
    centerrow = np.int16(np.ceil(nrow / 2) - 1)
    centercol = np.int16(np.ceil(ncol / 2) - 1)
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        num1 = np.round(((i - centerrow) * dv / radius) / du)
        (p1, p2) = np.int16(np.clip(np.sort(
            (-int(num1) + centercol, num1 + centercol)), 0, ncol - 1))
        mask[i, p1:p2 + 1] = np.ones(p2 - p1 + 1, dtype='float32')
    if drop < centerrow:
        mask[centerrow - drop:centerrow + drop + 1,
             :] = np.zeros((2 * drop + 1, ncol), dtype='float32')
    mask[:,centercol-1:centercol+2] = np.zeros((nrow, 3), dtype='float32')
    return mask


def find_center_dnn(tomo, theta, search_range, search_step=1, level=0, outpath='center', pad_length=0, **kwargs):

    rot_start, rot_end = search_range
    write_center(tomo[:, 0:1, :], theta, dpath=outpath,
                 cen_range=[rot_start / pow(2, level), rot_end / pow(2, level),
                            search_step / pow(2, level)],
                 pad_length=pad_length)
    return search_in_folder_dnn(outpath, **kwargs)


def search_in_folder_dnn(dest_folder, window=((600, 600), (1300, 1300)), dim_img=128, seed=1337, batch_size=50):

    patch_size = (dim_img, dim_img)
    nb_classes = 2
    save_intermediate = False
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    nb_evl = 100

    fnames = glob.glob(os.path.join(dest_folder, '*.tiff'))
    fnames = np.sort(fnames)

    mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

    weight_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    weight_path = os.path.join(weight_path, 'config', 'weight_center.h5')
    print(weight_path)
    mdl.load_weights(weight_path)
    start_time = time.time()
    Y_score = np.zeros((len(fnames)))

    for i in range(len(fnames)):
        print(fnames[i])
        img = dxchange.read_tiff(fnames[i])
        X_evl = np.zeros((nb_evl, dim_img, dim_img))

        for j in range(nb_evl):
            X_evl[j] = xlearn.img_window(img[window[0][0]:window[1][0], window[0][1]:window[1][1]], dim_img, reject_bg=True,
                                  threshold=1.5e-4, reset_random_seed=True, random_seed=j)
        # X_evl = convolve_stack(X_evl, get_gradient_kernel())
        X_evl = xlearn.convolve_stack(X_evl, xlearn.get_gradient_kernel())
        X_evl = xlearn.nor_data(X_evl)
        if save_intermediate:
            dxchange.write_tiff(X_evl, os.path.join('debug', 'x_evl', 'x_evl_{}'.format(i)), dtype='float32',
                                overwrite=True)
        X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)
        Y_evl = mdl.predict(X_evl, batch_size=batch_size)
        Y_score[i] = sum(np.dot(Y_evl, [0, 1]))

    ind_max = np.argmax(Y_score)
    best_center = float(os.path.splitext(os.path.basename(fnames[ind_max]))[0])
    print('Center search done in {} s. Optimal center is {}.'.format(time.time() - start_time, best_center))

    return best_center


def pad_sinogram(sino, length, mean_length=40, mode='edge'):

    assert sino.ndim == 3
    length = int(length)
    res = np.zeros([sino.shape[0], sino.shape[1], sino.shape[2] + length * 2])
    res[:, :, length:length+sino.shape[2]] = sino
    if mode == 'edge':
        for i in range(sino.shape[1]):
            mean_left = np.mean(sino[:, i, :mean_length], axis=1).reshape([sino.shape[0], 1])
            mean_right = np.mean(sino[:, i, -mean_length:], axis=1).reshape([sino.shape[0], 1])
            res[:, i, :length] = mean_left
            res[:, i, -length:] = mean_right

    return res


def write_center(tomo, theta, dpath='tmp/center', cen_range=None, pad_length=0):

    for center in np.arange(*cen_range):
        rec = tomopy.recon(tomo[:, 0:1, :], theta, algorithm='gridrec', center=center)
        if not pad_length == 0:
            rec = rec[:, pad_length:-pad_length, pad_length:-pad_length]
        dxchange.write_tiff(np.squeeze(rec), os.path.join(dpath, '{:.2f}'.format(center-pad_length)), overwrite=True)


def get_index(file_list, pattern=1):
    '''
    Get tile indices.
    :param file_list: list of files.
    :param pattern: pattern of naming. For files named with x_*_y_*, use
                    pattern=0. For files named with y_*_x_*, use pattern=1.
    :return:
    '''
    if pattern == 0:
        regex = re.compile(r".+_x(\d+)_y(\d+)(.*)")
        ind_buff = [m.group(1, 2) for l in file_list for m in [regex.search(l)] if m]
    elif pattern == 1:
        regex = re.compile(r".+_y(\d+)_x(\d+)(.*)")
        ind_buff = [m.group(2, 1) for l in file_list for m in [regex.search(l)] if m]
    return np.asarray(ind_buff).astype('int')


def start_file_grid(file_list, ver_dir=0, hor_dir=0, pattern=1):
    ind_list = get_index(file_list, pattern)
    if pattern == 0:
        x_max, y_max = ind_list.max(0)
        x_min, y_min = ind_list.min(0)
    elif pattern == 1:
        x_max, y_max = ind_list.max(0) + 1
        x_min, y_min = ind_list.min(0) + 1
    grid = np.empty((y_max, x_max), dtype=object)
    for k_file in range(len(file_list)):
        if pattern == 0:
            grid[ind_list[k_file, 1] - 1, ind_list[k_file, 0] - 1] = file_list[k_file]
        elif pattern == 1:
            grid[ind_list[k_file, 1], ind_list[k_file, 0]] = file_list[k_file]
    if ver_dir:
        grid = np.flipud(grid)
    if hor_dir:
        grid = np.fliplr(grid)
    return grid

