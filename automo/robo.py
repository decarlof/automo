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
import shutil
import sys
import re
import string
import argparse
import unicodedata
import ConfigParser
from os.path import expanduser
import h5py
import automo.util as util

from distutils.dir_util import mkpath
import logging

logger = logging.getLogger(__name__)
__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_process',
           'process']

class automo_exp:
    user_home = ''
    proc_dir = ''
    cf_file = ''
    cf = ''
    def_h5_fname = ''
    proc_list = ''
    macro_list = ''
    proc_folder = ''

class automo_robo:
    type = ''
    move = ''
    rename = ''
    proc_list = ''

def init():
    global exp
    exp = automo_exp()
    exp.user_home = expanduser("~")
    exp.proc_dir = os.path.join(exp.user_home, '.automo')

    exp.cf_file = os.path.join(exp.proc_dir, 'automo.ini')
    exp.cf = ConfigParser.ConfigParser()
    exp.cf.read(exp.cf_file)

    if exp.cf.has_option('settings', 'default_h5_fname'):
        exp.def_h5_fname = exp.cf.get('settings', 'default_h5_fname')

    # specify a different process folder
    if exp.cf.has_option('settings', 'python_proc_dir'):
        exp.proc_dir = exp.cf.get('settings', 'python_proc_dir')

    exp.proc_list = exp.cf.options('robos')
    exp.macro_list = [f for f in os.listdir(exp.proc_dir) if re.match(r'.+.py', f)]
    return exp


def process_folder(folder):
    """
    Create process list for all files in a folder

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.
    """


    exp = init()
    # files are sorted alphabetically
    exp.folder = folder
    files = [f for f in sorted(os.listdir(exp.folder)) if re.match(r'.+.h5', f)]

    os.chdir(exp.folder)

    for kfile in files:
        create_process(exp, kfile)

    return

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
    robo_att = get_robo_att(exp, robo_type)
    exec_process(fname, exp, robo_att) if robo_att
    return


def exec_process(folder, fname, robo_att):
    new_folder = robo_move(exp, fname, robo_att.move)
    os.chdir(new_folder)

    new_fname = robo_rename(exp, fname, robo_att.rename)

    robo_process(exp, new_fname, robo_att.proc_list)

    os.chdir(folder)
    return

def get_robo_att(exp, robo_type):
    global robo_att
    robo_att = automo_robo()
    if exp.cf.has_option('robos', robo_type):
        robo_att.type = robo_type
        processes = exp.cf.get('robos', robo_type)
        robo_att.proc_list = processes.split(', ')
        if exp.cf.has_option('robos_move', robo_type):
            robo_att.move = exp.cf.get('robos_move', robo_type)
        else:
            robo_att.move = new_folder
        if exp.cf.has_option('robos_rename', robo_type):
            robo_att.rename = exp.cf.get('robos_rename', robo_type)
        else:
            robo_att.rename = False
    else:
        print('Robo type not found!!!')
        print('Doing nothing.')
        robo_att = None
    return robo_att

def get_file_name(file):
    basename = os.path.splitext(file)[0]
    return basename

def robo_move(exp, file, move_type):
    basename = get_file_name(file)
    if move_type=='new_folder':
        os.mkdir(basename)
        shutil.move (file, os.path.join(basename,file))
    elif move_type=='same_folder':
        print 'not implemented'
    else:
        print 'not implemented'
    return basename

def robo_rename(file, rename_type):
    if rename_type:
        os.rename(file,exp.def_h5_fname)
        return exp.def_h5_fname
    else:
        return file

def robo_process(exp, file, proc_list):
    for proc in proc_list:
        runtime_line = "python " + os.path.join(exp.proc_dir, proc)+ ".py " + file + " -1 -1 -1 -1"
        print runtime_line
        #os.system(runtime_line)


if __name__ == "__main__":
    process_folder(sys.argv[1])
