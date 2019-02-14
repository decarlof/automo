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
from scipy.ndimage import fourier_shift
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d
from tomopy import find_center_pc
import dxchange
import operator
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


# logger = logging.getLogger(__name__)
PI = 3.1415927

__author__ = ['Francesco De Carlo', 'Ming Du','Rafael Vescovi']
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['append',
           'clean_entry',
           'clean_folder_name', 
           'dataset_info',
           'try_folder',
           'h5group_dims',
           'touch',
           'write_first_frames',
           'find_center_com']


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

    if not (abs(theta[-1] - theta[0] - 2 * np.pi) < 0.1 or abs(theta[-1] - theta[0] - np.pi) < 0.1):
        warnings.warn('There might be a problem in theta. Double check the values.')
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


def get_histogram(img, bin_min, bin_max, n_bin=256):

    bins = np.linspace(bin_min, bin_max, n_bin)
    counts = np.zeros(n_bin+1)
    ind = np.squeeze(np.searchsorted(bins, img))
    for i in ind:
        counts[i] += 1
    return counts / img.size


def equalize_histogram(img, bin_min, bin_max, n_bin=256):

    histogram = get_histogram(img, bin_min, bin_max, n_bin=n_bin)
    bins = np.linspace(bin_min, bin_max, n_bin)
    e_table = np.zeros(n_bin + 1)
    res = np.zeros(img.shape)
    s_max = float(np.max(img))
    for i in range(bins.size):
        e_table[i] = s_max * np.sum(histogram[:i+1])
    ind = np.searchsorted(bins, img)
    for (y, x), i in np.ndenumerate(ind):
        res[y, x] = e_table[i]
    return res


def sino_360_to_180(data, overlap=0, rotation='right', blend=True):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded.
    Parameters
    ----------
    data : ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.
    Returns
    -------
    ndarray
        Output 3D data.
    """
    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))

    lo = overlap//2
    ro = overlap - lo
    n = dx//2

    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)

    if blend:
        if rotation == 'left':
            img1 = data[n:2*n, :, ro:][:, :, ::-1]
            img2 = data[:n, :, :]
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            shift = [0, dz - lo]
            for i in range(out.shape[1]):
                out[:, i, :] = np.fliplr(img_merge_pyramid(img2[:, i, :], img1[:, i, :], shift=shift, depth=2))
        elif rotation == 'right':
            img1 = data[:n, :, :]
            img2 = data[n:2*n, :, :-ro][:, :, ::-1]
            shift = [0, dz-lo]
            for i in range(out.shape[1]):
                out[:, i, :] = img_merge_pyramid(img1[:, i, :], img2[:, i, :], shift=shift, depth=2)
    else:
        if rotation == 'left':
            out[:, :, -(dz-lo):] = data[:n, :, lo:]
            out[:, :, :-(dz-lo)] = data[n:2*n, :, ro:][:, :, ::-1]
        elif rotation == 'right':
            out[:, :, :dz-lo] = data[:n, :, :-lo]
            out[:, :, dz-lo:] = data[n:2*n, :, :-ro][:, :, ::-1]

    return out


def img_merge_pyramid(img1, img2, shift, margin=100, blur=0.4, depth=5):
    """
    Perform pyramid blending. Codes are adapted from Computer Vision Lab, Image blending using pyramid,
    https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/.
    Users are strongly suggested to run tests before beginning the actual stitching job using this function to determine
    the biggest depth value that does not give seams due to over-blurring.
    """

    t00 = time.time()
    t0 = time.time()
    # print(    'Starting pyramid blend...')
    newimg, img2 = arrange_image(img1, img2, shift)
    if abs(shift[0]) < margin and abs(shift[1]) < margin:
        return newimg
    # print('    Blend: Image aligned and built in', str(time.time() - t0))

    t0 = time.time()
    case, rough_shift, corner, buffer1, buffer2, wid_hor, wid_ver = find_overlap(img1, img2, shift, margin=margin)
    if case == 'skip':
        return newimg
    mask2 = np.ones(buffer1.shape)
    if abs(rough_shift[1]) > margin:
        mask2[:, :int(wid_hor / 2)] = 0
    if abs(rough_shift[0]) > margin:
        mask2[:int(wid_ver / 2), :] = 0
    ##
    buffer1[np.isnan(buffer1)] = 0
    mask2[np.isnan(mask2)] = 1
    t0 = time.time()
    gauss_mask = _gauss_pyramid(mask2.astype('float'), depth, blur, mask=True)
    gauss1 = _gauss_pyramid(buffer1, depth, blur)
    gauss2 = _gauss_pyramid(buffer2, depth, blur)
    lapl1 = _lapl_pyramid(gauss1, blur)
    lapl2 = _lapl_pyramid(gauss2, blur)
    ovlp_blended = _collapse(_blend(lapl2, lapl1, gauss_mask), blur)
    # print('    Blend: Blending done in', str(time.time() - t0), 'sec.')

    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + mask2.shape[1]] = \
            ovlp_blended[:wid_ver, :]
        newimg[corner[0, 0] + wid_ver:corner[0, 0] + mask2.shape[0], corner[0, 1]:corner[0, 1] + wid_hor] = \
            ovlp_blended[wid_ver:, :wid_hor]
    else:
        newimg[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor] = ovlp_blended
    # print('    Blend: Done with this tile in', str(time.time() - t00), 'sec.')
    gc.collect()

    return newimg


def _generating_kernel(a):
    w_1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(w_1d, w_1d)


def _ireduce(image, blur):
    kernel = _generating_kernel(blur)
    outimage = convolve2d(image, kernel, mode='same', boundary='symmetric')
    out = outimage[::2, ::2]
    return out


def _iexpand(image, blur):
    kernel = _generating_kernel(blur)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4 * convolve2d(outimage, kernel, mode='same', boundary='symmetric')
    return out


def _gauss_pyramid(image, levels, blur, mask=False):
    output = []
    if mask:
        image = gaussian_filter(image, 20)
    output.append(image)
    tmp = np.copy(image)
    for i in range(0, levels):
        tmp = _ireduce(tmp, blur)
        output.append(tmp)
    return output


def _lapl_pyramid(gauss_pyr, blur):
    output = []
    k = len(gauss_pyr)
    for i in range(0, k - 1):
        gu = gauss_pyr[i]
        egu = _iexpand(gauss_pyr[i + 1], blur)
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output


def _blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
    blended_pyr = []
    k = len(gauss_pyr_mask)
    for i in range(0, k):
        p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
        p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
        blended_pyr.append(p1 + p2)
    return blended_pyr


def _collapse(lapl_pyr, blur):
    output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr) - 1, 0, -1):
        lap = _iexpand(lapl_pyr[i], blur)
        lapb = lapl_pyr[i - 1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output


def arrange_image(img1, img2, shift, order=1, trim=True):
    """
    Place properly aligned image in buff

    Parameters
    ----------
    img1 : ndarray
        Substrate image array.

    img2 : ndarray
        Image being added on.

    shift : float
        Subpixel shift.
    order : int
        Order that images are arranged. If order is 1, img1 is written first and img2 is placed on the top. If order is
        2, img2 is written first and img1 is placed on the top.
    trim : bool
        In the case that shifts involve negative or float numbers where Fourier shift is needed, remove the circular
        shift stripe.

    Returns
    -------
    newimg : ndarray
        Output array.
    """
    rough_shift = get_roughshift(shift).astype('int')
    adj_shift = shift - rough_shift.astype('float')
    if np.count_nonzero(np.isnan(img2)) > 0:
        int_shift = np.round(adj_shift).astype('int')
        img2 = np.roll(np.roll(img2, int_shift[0], axis=0), int_shift[1], axis=1)
    else:
        img2 = realign_image(img2, adj_shift)
    if trim:
        temp = np.zeros(img2.shape-np.ceil(np.abs(adj_shift)).astype('int'))
        temp[:, :] = img2[:temp.shape[0], :temp.shape[1]]
        img2 = np.copy(temp)
        temp = 0
    # new_shape = map(int, map(max, map(operator.add, img2.shape, rough_shift), img1.shape))
    new_shape = np.array(np.array(img2.shape) + np.array(rough_shift))
    new_shape = np.max(np.array([new_shape, np.array(img1.shape)]), axis=0).astype('int')
    newimg = np.empty(new_shape)
    newimg[:, :] = np.NaN
    if order == 1:
        newimg[0:img1.shape[0], 0:img1.shape[1]] = img1
        notnan = np.isfinite(img2)
        newimg[rough_shift[0]:rough_shift[0] + img2.shape[0], rough_shift[1]:rough_shift[1] + img2.shape[1]][notnan] \
            = img2[notnan]
    elif order == 2:
        newimg[rough_shift[0]:rough_shift[0] + img2.shape[0], rough_shift[1]:rough_shift[1] + img2.shape[1]] = img2
        notnan = np.isfinite(img1)
        newimg[0:img1.shape[0], 0:img1.shape[1]][notnan] = img1[notnan]
    else:
        print('Warning: images are not arranged due to misspecified order.')
    gc.collect()
    if trim:
        return newimg, img2
    else:
        return newimg


def get_roughshift(shift):

    rough_shift = np.ceil(shift)
    rough_shift[rough_shift < 0] = 0
    return rough_shift


def realign_image(arr, shift, angle=0):
    """
    Translate and rotate image via Fourier

    Parameters
    ----------
    arr : ndarray
        Image array.

    shift: float
        Mininum and maximum values to rescale data.

    angle: float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.count_nonzero(np.abs(np.array(shift) - np.round(shift)) < 0.01) == 2:
        temp = np.roll(arr, int(shift[0]), axis=0)
        temp = np.roll(temp, int(shift[1]), axis=1)
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp


def find_overlap(img1, img2, shift, margin=50):

    rough_shift = get_roughshift(shift)
    corner = _get_corner(rough_shift, img2.shape)
    if min(img1.shape) < margin or min(img2.shape) < margin:
        return 'skip', rough_shift, corner, None, None, None, None
    if abs(rough_shift[1]) > margin and abs(rough_shift[0]) > margin:
        abs_width = np.count_nonzero(np.isfinite(img1[-margin, :]))
        abs_height = np.count_nonzero(np.isfinite(img1[:, abs_width - margin]))
        temp0 = img2.shape[0] if corner[1, 0] <= abs_height - 1 else abs_height - corner[0, 0]
        temp1 = img2.shape[1] if corner[1, 1] <= img1.shape[1] - 1 else img1.shape[1] - corner[0, 1]
        mask = np.zeros([temp0, temp1], dtype='bool')
        temp = img1[corner[0, 0]:corner[0, 0] + temp0, corner[0, 1]:corner[0, 1] + temp1]
        temp = np.isfinite(temp)
        wid_ver = np.count_nonzero(temp[:, -1])
        wid_hor = np.count_nonzero(temp[-1, :])
        mask[:wid_ver, :] = True
        mask[:, :wid_hor] = True
        buffer1 = img1[corner[0, 0]:corner[0, 0] + mask.shape[0], corner[0, 1]:corner[0, 1] + mask.shape[1]]
        buffer2 = img2[:mask.shape[0], :mask.shape[1]]
        #buffer1[np.invert(mask)] = np.nan
        #buffer2[np.invert(mask)] = np.nan
        case = 'tl'
        if abs_width < corner[0, 1]:
            case = 'skip'
    # for new image with overlap at top only
    elif abs(rough_shift[1]) < margin and abs(rough_shift[0]) > margin:
        abs_height = np.count_nonzero(np.isfinite(img1[:, margin]))
        wid_ver = abs_height - corner[0, 0]
        wid_hor = img2.shape[1] if img1.shape[1] > img2.shape[1] else img2.shape[1] - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        case = 't'
    # for new image with overlap at left only
    else:
        abs_width = np.count_nonzero(np.isfinite(img1[margin, :]))
        wid_ver = img2.shape[0] - corner[0, 0]
        wid_hor = abs_width - corner[0, 1]
        buffer1 = img1[corner[0, 0]:corner[0, 0] + wid_ver, corner[0, 1]:corner[0, 1] + wid_hor]
        buffer2 = img2[:wid_ver, :wid_hor]
        case = 'l'
        if abs_width < corner[0, 1]:
            case = 'skip'
    res1 = np.copy(buffer1)
    res2 = np.copy(buffer2)
    return case, rough_shift, corner, res1, res2, wid_hor, wid_ver


def _get_corner(shift, img2_shape):
    corner_uly, corner_ulx, corner_bry, corner_brx = (shift[0], shift[1], shift[0] + img2_shape[0] - 1,
                                                      shift[1] + img2_shape[1] - 1)
    return np.squeeze([[corner_uly, corner_ulx], [corner_bry, corner_brx]]).astype('int')


def preprocess(dat, blur=None, normalize_bg=False):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    if normalize_bg:
        dat = tomopy.normalize_bg(dat)
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat


def write_first_frames(folder='.', data_format='aps_32id'):

    flist = glob.glob(os.path.join(folder, '*.h5'))
    for f in flist:
        print(f)
        dat, flt, drk, _ = read_data_adaptive(os.path.join(folder, f), proj=(0, 1), data_format=data_format)
        dat = tomopy.normalize(dat, flt, drk)
        f = os.path.splitext(os.path.basename(f))[0]
        dxchange.write_tiff(dat, os.path.join('first_frames', f), dtype='float32', overwrite=True)


def find_center_com(sino, return_com_list=False):

    sino = np.squeeze(sino)
    line_com_ls = []
    for i, line in enumerate(sino):
        line_int = np.sum(line)
        com = np.sum(np.arange(sino.shape[1]) * line) / line_int
        line_com_ls.append(com)
    if return_com_list:
        return (np.mean(line_com_ls), line_com_ls)
    else:
        return np.mean(line_com_ls)