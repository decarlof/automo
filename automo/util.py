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

import os
import sys
import h5py
import string
import argparse
import unicodedata
import ConfigParser
from os.path import expanduser
import dxchange.reader as dxreader

from distutils.dir_util import mkpath

__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['dataset_dims']


    

def read_hdf5_dims(fname, dataset):
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
        tomo = read_hdf5_dims(fname, tomo_grp)
        flat = read_hdf5_dims(fname, flat_grp)
        dark = read_hdf5_dims(fname, dark_grp)
        theta = read_hdf5_dims(fname, theta_grp)
        theta_flat = read_hdf5_dims(fname, theta_flat_grp)
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


def dataset_dims(fname, img_type = 'data_white'):
    """
    Determine the tomographic data set info    

    Parameters
    ----------
    fname : str
        h5 full path file name.


    Returns
    -------
    info : list
        List containing the data set array info.
    """
    array = img_type
    exchange_base = "exchange"
    flat_grp = '/'.join([exchange_base, array])

    try: 
        flat = dxreader.read_hdf5(fname, flat_grp)
        return flat.shape
        
    except OSError:
        pass


