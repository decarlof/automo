
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo example to generage sample previews.
"""

from __future__ import print_function

import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob
import re, shutil

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="existing recon foldername")
    parser.add_argument("shift", help="shift between datasets",default='auto')
    parser.add_argument("new_folder",help="target folder",default='auto')
    args = parser.parse_args()


    prefix = args.prefix
    shift = args.shift
    new_folder = args.new_folder

    folder_list = glob(prefix)
    folder_list.sort()

    folder_grid = util.start_file_grid(folder_list, pattern=1)

    if new_folder == 'auto':
        new_folder = prefix + '_restack'

    try:
        os.makedirs(new_folder)
    except:
        pass

    if shift == 'auto':
        try:
            f = open(os.path.join(new_folder, 'shift.txt'), 'w')
            shift_ls = f.readlines()
            shift_ls = map(int, shift_ls)
            f.close()
        except:
            raise IOError('I could not find shift.txt in the target folder.')
    else:
        shift_ls = [int(shift)] * (len(folder_list) - 1) #number of slices to keep
    
    accum = 0
    for i, folder in enumerate(folder_grid[:, 0]):
        shift = shift_ls[i]
        file_list = glob(os.path.join(os.path.join(folder, 'recon', 'recon*.tiff')))
        file_list.sort()
        if i < len(folder_list) - 1:
            for j, f in enumerate(file_list[:shift]):
                shutil.copyfile(f, os.path.join('full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
        else:
            for j, f in enumerate(file_list):
                shutil.copyfile(f, os.path.join('full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
        accum += shift

if __name__ == "__main__":
    main(sys.argv[1:])