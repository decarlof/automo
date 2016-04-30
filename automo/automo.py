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
Module to create basic tomo directory structure.

"""

import os
import sys
import string
import argparse
import unicodedata

from distutils.dir_util import mkpath

__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_cmd',
           'create_tomo']


def create_cmd_old(folder):
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
    
    # files are sorted alphabetically
    files = sorted(os.listdir(folder))
    cmd = []
    try:
        for fname in files:
            sname = fname.split('.')
            try:
                ext = sname[1]
                if ext == "h5":
                    for subfolder in subfolders:
                        cmd.append("mkdir " + sys.argv[1] + sname[0] + os.sep + subfolder)
                    cmd.append("mv " + sys.argv[1] + fname + " " + sys.argv[1] + sname[0] + os.sep + h5_fname)
            except: # does not have an extension
                if os.path.isdir(folder + fname): # is a folder?
                    for process in processes:
                        cmd.append("python " + process + ".py " + folder + fname + os.sep)                     
                pass    
        return cmd
    except OSError:
        pass
    
def create_cmd(folder):
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
    
    # files are sorted alphabetically
    files = sorted(os.listdir(folder))
    cmd = []
    try:
        for fname in files:
            sname = fname.split('.')
            try:
                ext = sname[1]
                if ext == "h5":
                    cmd.append("mkdir " + sys.argv[1] + sname[0] + os.sep)
                    cmd.append("mv " + sys.argv[1] + fname + " " + sys.argv[1] + sname[0] + os.sep + h5_fname)
            except: # does not have an extension
                if os.path.isdir(folder + fname): # is a folder?
                    for process in processes:
                        cmd.append("python " + process + ".py " + folder + fname + os.sep)                     
                pass    
        return cmd
    except OSError:
        pass


def create_tomo(argv):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="new or existing folder")
    args = parser.parse_args()
    
    try: 
        folder = os.path.normpath(_clean_folder_name(args.folder)) + os.sep # will add the trailing slash if it's not already there.
        if _try_folder(folder):
            cmd_list = create_cmd(folder)
            for cmd in cmd_list:
                print cmd
                os.system(cmd)
    except: 
        pass


def _clean_folder_name(directory):
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

    valid_folder_name_chars = "-_"+ os.sep + "%s%s" % (string.ascii_letters, string.digits)
    cleaned_folder_name = unicodedata.normalize('NFKD', directory.decode('utf-8', 'ignore')).encode('ASCII', 'ignore')
    
    return ''.join(c for c in cleaned_folder_name if c in valid_folder_name_chars)


def _try_folder(directory):
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
            print directory + " exists"
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



