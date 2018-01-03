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
from tqdm import tqdm
import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="existing recon foldername")
    parser.add_argument("--shift", help="shift between datasets",default='auto')
    parser.add_argument("--new_folder",help="target folder",default='auto')
    args = parser.parse_args()


    prefix = args.prefix
    shift = args.shift
    new_folder = args.new_folder


    folder_list = sorted(glob(prefix+'*[!restack]'))

    folder_grid = util.start_file_grid(folder_list, pattern=1)

    if new_folder == 'auto':
        new_folder = prefix + '_restack'


    print (new_folder)
    try:
        os.makedirs(new_folder)
    except:
        pass

    if shift == 'auto':
        try:
            f = open(os.path.join(new_folder, 'shift.txt'), 'r')
            shift_ls = f.readlines()
            shift_ls = [int(float(i)) for i in shift_ls]
            f.close()
        except:
            raise IOError('I could not find shift.txt in the target folder.')
    else:
        shift_ls = [int(shift)] * (len(folder_list) - 1) #number of slices to keep

    os.makedirs(os.path.join(new_folder, 'full_stack'))


#find_size

    max_size_x = 0
    max_size_y = 0
    for i, folder in enumerate(folder_grid[:, 0]):
        file_list = sorted(glob(os.path.join(os.path.join(folder, 'recon', 'recon*.tiff'))))
        file_list.sort()
        slice1 = dxchange.read_tiff(file_list[0])
        max_size_x = max(max_size_x, slice1.shape[0])
        max_size_y = max(max_size_y, slice1.shape[1]) # it is usually a square but whatever.
 
    new_slice1 = np.zeros([max_size_x, max_size_y])

    accum = 0
    for i, folder in tqdm(enumerate(folder_grid[:, 0])):
        if i < folder_grid.shape[0] - 1:
            shift = shift_ls[i]
        file_list = glob(os.path.join(os.path.join(folder, 'recon', 'recon*.tiff')))
        file_list.sort()
        if i < folder_grid.shape[0] - 1:
            for j, f in enumerate(file_list[:shift]):
                slice1 = dxchange.read_tiff(file_list[j])
                (margin_y, margin_x) = ((np.array([max_size_y, max_size_x]) - np.array(slice1.shape)) / 2).astype('int')
                new_slice1[margin_y:margin_y+slice1.shape[0], margin_x:margin_x+slice1.shape[1]] = slice1
                new_slice1 = new_slice1.astype('float32')
                dxchange.write_tiff(new_slice1, os.path.join(new_folder, 'full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
                #shutil.copyfile(f, os.path.join(new_folder, 'full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
        else:
            for j, f in enumerate(file_list):
                slice1 = dxchange.read_tiff(file_list[j])
                (margin_y, margin_x) = ((np.array([max_size_y, max_size_x]) - np.array(slice1.shape)) / 2).astype('int')
                new_slice1[margin_y:margin_y+slice1.shape[0], margin_x:margin_x+slice1.shape[1]] = slice1
                new_slice1 = new_slice1.astype('float32')
                dxchange.write_tiff(new_slice1, os.path.join(new_folder, 'full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
                #shutil.copyfile(f, os.path.join(new_folder, 'full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
        accum += shift

if __name__ == "__main__":
    main(sys.argv[1:])