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
import string
import argparse
import unicodedata
import ConfigParser
from os.path import expanduser

import automo import util

from distutils.dir_util import mkpath

__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_move',
           'create_test',
           'move',
           'test']


def init():
    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    pdir = cf.get('settings', 'python_proc_dir')
    processes = cf.get('settings', 'python_proc')
    processes = processes.split(', ')
    default_h5_fname = cf.get('settings', 'default_h5_fname')
    default_proc_fname = cf.get('settings', 'default_proc_fname')


def create_move(folder):
    """
    Create a list of commands to move all user_selected_name.h5 files in 
    folder to folder/user_selected_name/data.h5
    

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.


    Returns
    -------
    cmd : list
        List of mkdir and mv commands.
    """
    
    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    default_h5_fname = cf.get('settings', 'default_h5_fname')
    
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
                    cmd.append("mv " + sys.argv[1] + fname + " " + sys.argv[1] + sname[0] + os.sep + default_h5_fname)
            except: # does not have an extension
                pass
        return cmd
    except OSError:
        pass


def move(argv):
    """
    Move all user_selected_name.h5 files in the folder 
    (passed as argv) to folder/user_selected_name/data.h5    

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.


    Returns
    -------
    cmd : list
        List of mkdir and mv commands.
    """
   
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="new or existing folder")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    default_proc_fname = cf.get('settings', 'default_proc_fname')

    try: 
        folder = os.path.normpath(util.clean_folder_name(args.folder)) + os.sep # will add the trailing slash if it's not already there.
        if util.try_folder(folder):
            cmd_list = create_move(folder)
            for cmd in cmd_list:
                print cmd
                os.system(cmd)
                cmd = '\n' + cmd
                util.append(folder + default_proc_fname, cmd)
        print("-----------------------------------------------------------")
    except: 
        pass


def create_test(folder):
    """
    Create a list of commands to run a set of default functions 
    on data.h5 files located in folder/user_selected_name/data.h5
    

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.

    """
    
    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    pdir = cf.get('settings', 'python_proc_dir')
    processes = cf.get('settings', 'python_proc')
    processes = processes.split(', ')

    default_h5_fname = cf.get('settings', 'default_h5_fname')

    # files are sorted alphabetically
    files = sorted(os.listdir(folder))
    cmd = []

    try:
        for fname in files:
            sname = fname.split('.')
            try:
                ext = sname[1]
            except: # does not have an extension
                if os.path.isdir(folder + fname): # is a folder?
                    for process in processes:
                        cmd.append("python " + pdir + process + ".py " + folder + fname + os.sep + default_h5_fname + " -1 -1 -1 -1")   
                pass
        return cmd
    except OSError:
        pass


def test(argv):
    """
    Execute all test listed in the ~/.tomo folder

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="new or existing folder")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    default_proc_fname = cf.get('settings', 'default_proc_fname')

    try: 
        folder = os.path.normpath(util.clean_folder_name(args.folder)) + os.sep # will add the trailing slash if it's not already there.
        if util.try_folder(folder):
            cmd_list = create_test(folder)
            for cmd in cmd_list:
                print cmd
                os.system(cmd)
                cmd1 = '\n' + cmd
                util.append(folder + default_proc_fname, cmd1)
        print("-----------------------------------------------------------")
    except: 
        pass


