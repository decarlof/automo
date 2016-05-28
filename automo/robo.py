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
import glob
import sys
import re
import string
import argparse
import unicodedata
import ConfigParser
from os.path import expanduser

import automo.util as util

from distutils.dir_util import mkpath
import logging

logger = logging.getLogger(__name__)
__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_process',
           'process']


def init():
    user_home = expanduser("~")
    proc_dir = os.path.join(user_home, '.automo')

    cf_file = os.path.join(proc_dir, 'automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(cf_file)

    if cf.has_option('settings', 'default_h5_fname'):
        default_h5_fname = cf.get('settings', 'default_h5_fname')

    # specify a different process folder
    if cf.has_option('settings', 'python_proc_dir'):
        proc_dir = cf.get('settings', 'python_proc_dir')

    proc_list = cf.options('robos')
    macro_list = os.listdir(proc_dir)
    macro_list = [f for f in os.listdir(proc_dir) if re.match(r'.+.py', f)]



def process_folder(folder):
    """
    Create process list for all files in a folder

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.

    """

    # files are sorted alphabetically
    files = [f for f in sorted(os.listdir(folder)) if re.match(r'.+.h5', f)]

    for kfile in files:
        create_process(file)


def create_process(file):
    """
    Create a list of commands to run a set of default functions
    on .h5 files located in folder/user_selected_name/data.h5


    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.

    """
    robo_type = 'tomo'
    robo_att = get_robo_att(robo_type)
    if robo_att exec_process(folder, fname, robo_att)
return


def exec_process(folder, fname, robo_att):

    folder, fname = robo_move(folder,fname,robo_att.move)
    folder, fname = robo_rename(folder,fname,robo_att.rename)
    folder, fname = robo_process(folder, fname, robo_att.processes)
    return

def get_robo_att(robo_type):
    if cf.has_option('robos', robo_type):
        robo_att.robo_type = robo_type
        processes = cf.get('robos', robo_type)
        robo_att.processes = processes.split(', ')
        if cf.has_option('robos_move', robo_type):
            robo_att.move = cf.get('robos_move', robo_type)
        else:
            robo_att.move = new_folder

        if cf.has_option('robos_rename', robo_type):
            robo_att.rename = cf.get('robos_rename', robo_type)
        else:
            robo_att.rename = False
    else:
        print('Robo type not found!!!')
        print('Doing nothing.')
        robo_att = None
    return robo_att

def robo_move(folder, file, move_type):
    print 'batatas'

def robo_rename(file, rename_type):
    print 'batatas'

def robo_process(file, proc_list):
    for cmd in cmd_list:
        print cmd
        #os.system(cmd)
        cmd1 = '\n' + cmd
        util.append(folder + default_proc_fname, cmd1)
