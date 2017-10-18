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
import subprocess

from distutils.dir_util import mkpath
import logging

logger = logging.getLogger(__name__)
__author__ = "Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['process_folder']

class automo_exp:
    user_home = ''
    proc_dir = ''
    cf_file = ''
    cf = ''
    def_h5_fname = ''
    proc_list = ''
    macro_list = ''
    proc_folder = ''
    log_file = ''
    log_name = ''

class automo_robo:
    type = ''
    move = ''
    rename = ''
    proc_list = ''

def init(ini_name='automo.ini'):
    global exp
    exp = automo_exp()
    exp.user_home = expanduser("~")
    exp.proc_dir = os.path.join(exp.user_home, '.automo')

    exp.cf_file = os.path.join(exp.proc_dir, ini_name)
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


def process_folder(folder, ini_name='automo.ini', check_usage=True, **kwargs):
    """
    Create process list for all files in a folder

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.
    kwargs : dictionaries containing input options. One dictionary per keyword.
             E.g.: process_folder(folder, preview=preview_dict, recon=recon_dict)
             where preview_dict = {'proj_st':0, 'proj_end':1, 'proj_step':1,
                                   'slice_st':500, 'slice_end':501, 'slice_step':1}, etc.
    """

    exp = init(ini_name=ini_name)
    # files are sorted alphabetically
    exp.folder = folder

    exp.log = folder + exp.log_name

    files = [f for f in sorted(os.listdir(exp.folder)) if re.match(r'.+.h5', f)]

    os.chdir(exp.folder)

    tomosaic_naming = '.+_[x,y]\d+\..+'

    # option_dict = classify_kwargs(exp, **kwargs)

    for kfile in files:
        if '_180_' in kfile:
            robo_type = 'tomo_180'
        elif '_360_' in kfile:
            robo_type = 'tomo_360'
        elif re.match(tomosaic_naming, kfile):
            robo_type = 'tomosaic'
        else:
            robo_type = 'std'
        if check_usage:
            ret = str(subprocess.check_output('lsof'))
            if kfile not in ret:
                create_process(exp, kfile, robo_type=robo_type, check_usage=check_usage, **kwargs)
            else:
                print('{:s} skipped because it is currently in use.'.format(kfile))
        else:
            create_process(exp, kfile, robo_type=robo_type, check_usage=check_usage, **kwargs)

    return

def create_process(exp, file, robo_type='tomo', **kwargs):
    """
    Create a list of commands to run a set of default functions
    on .h5 files located in folder/user_selected_name/data.h5

    Parameters
    ----------
    folder : str
        Folder containing multiple h5 files.
    """
    robo_att = get_robo_att(exp, robo_type)
    if robo_att:
        exec_process(exp, file, robo_att, **kwargs)
    return


def exec_process(exp, fname, robo_att, **kwargs):
    new_folder = robo_move(exp, fname, robo_att.move)
    os.chdir(new_folder)

    new_fname = robo_rename(exp, fname, robo_att.rename)

    robo_process(exp, new_fname, robo_att.proc_list, **kwargs)

    os.chdir(exp.folder)
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
            robo_att.move = exp.new_folder
        if exp.cf.has_option('robos_rename', robo_type):
            robo_att.rename = True if (exp.cf.get('robos_rename', robo_type) == True) else False
        else:
            robo_att.rename = False
    else:
        print('Robo type not found!!!')
        print('Doing nothing.')
        robo_att = None
    return robo_att

def get_file_name(file):

    print(file)
    basename = os.path.splitext(file)[0]
    return basename

def robo_move(exp, file, move_type):

    basename = get_file_name(file)
    if move_type=='new_folder':
        basename = get_file_name(file)
        if ~os.path.exists(basename):
            os.mkdir(basename)
        shutil.move(file, os.path.join(basename,file))
    elif move_type == 'existing_folder':
        regex = re.compile(r"(.+)_y(\d+)_x(\d+).+")
        reg_dict = regex.search(file)
        if reg_dict is not None:
            basename = reg_dict.group(1)
        else:
            basename = get_file_name(file)
        try:
            os.mkdir(basename)
        except:
            pass
        shutil.move(file, os.path.join(basename,file))
    else:
        print('not implemented')
    return basename

def robo_rename(exp, file, rename_type):

    if rename_type:
        os.rename(file, exp.def_h5_fname)
        return exp.def_h5_fname
    else:
        return file

def robo_process(exp, file, proc_list, **kwargs):

    log = open('recon.sh', 'w')
    for proc in proc_list:
        if proc == 'preview':
            opts = [kwargs['preview']['proj_st'], kwargs['preview']['proj_end'], kwargs['preview']['proj_step'],
                    kwargs['preview']['slice_st'], kwargs['preview']['slice_end'], kwargs['preview']['slice_step']]
        elif proc == 'center':
            opts = [kwargs['center']['rot_start'], kwargs['center']['rot_end'], kwargs['center']['rot_step'],
                    kwargs['center']['slice'], kwargs['center']['n_slice'],
                    kwargs['center']['medfilt_size'], kwargs['center']['level']]
        elif proc == 'recon':
            opts = [kwargs['recon']['center_folder'], kwargs['recon']['sino_start'], kwargs['recon']['sino_end'],
                    kwargs['recon']['sino_step'], kwargs['recon']['medfilt_size'], kwargs['recon']['level'],
                    kwargs['recon']['chunk_size']]
        elif proc == 'preview_360':
            opts = [kwargs['preview']['slice_st'], kwargs['preview']['slice_end'], kwargs['preview']['slice_step']]
        elif proc == 'center_360':
            opts = [kwargs['center_360']['rot_start'], kwargs['center_360']['rot_end'], kwargs['center_360']['rot_step'],
                    kwargs['center_360']['slice'], kwargs['center_360']['n_slice'],
                    kwargs['center_360']['medfilt_size'], kwargs['center_360']['level']]
        elif proc == 'recon_360':
            opts = [kwargs['recon_360']['center_folder'], kwargs['recon_360']['sino_start'], kwargs['recon_360']['sino_end'],
                    kwargs['recon_360']['sino_step'], kwargs['recon_360']['medfilt_size'], kwargs['recon_360']['level'],
                    kwargs['recon_360']['chunk_size']]
        elif proc == 'preview_tomosaic':
            opts = [kwargs['preview']['proj_st'], kwargs['preview']['proj_end'], kwargs['preview']['proj_step'],
                    kwargs['preview']['slice_st'], kwargs['preview']['slice_end'], kwargs['preview']['slice_step'],
                    kwargs['preview']['write_aux']]
        opts = ' '.join(map(str, opts))
        opts = ' ' + opts
        runtime_line = 'python ' + os.path.join(exp.proc_dir, proc)+ '.py ' + file + opts
        print(runtime_line)
        # log.write(runtime_line + '\n')
        if 'recon' in proc:
            log.write('python /local/Software/rchard/automo/config/recon.py ' + file + opts + '\n')
        os.system(runtime_line)
    log.close()


if __name__ == "__main__":
    process_folder(sys.argv[1])