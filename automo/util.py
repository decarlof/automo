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
__all__ = ['append',
           'clean_entry',
           'clean_folder_name', 
           'dataset_info',
           'try_folder',
           'h5group_dims',
           'touch']


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

def clean_folder_name(directory):    """    Clean the folder name from unsupported characters before    creating it.        Parameters    ----------    folder : str        Folder that will be containing multiple h5 files.    """    valid_folder_name_chars = "-_"+ os.sep + "%s%s" % (string.ascii_letters, string.digits)    cleaned_folder_name = unicodedata.normalize('NFKD', directory.decode('utf-8', 'ignore')).encode('ASCII', 'ignore')        return ''.join(c for c in cleaned_folder_name if c in valid_folder_name_chars)
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

    try:
        if os.path.isdir(directory):
            return True
        else:
            print directory + " does not exist"
            a = raw_input('Would you like to create ' + directory + ' ? ').lower()
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
    with open(path, 'a'):        os.utime(path, None)


def append(fname, process):
    with open(fname, "a") as pfile:        pfile.write(process)


