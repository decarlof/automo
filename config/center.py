#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function

import argparse
import os
import sys
import re
from os.path import expanduser
import warnings

import h5py
import dxchange
import tomopy
import numpy as np
from glob import glob
from tqdm import tqdm

import automo.util as util

def debug_print(debug, text):
    if debug:
        print(text)

def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("--rot_start", help="rotation axis start location",default=860)
    parser.add_argument("--rot_end", help="rotation axis end location",default=1060)
    parser.add_argument("--rot_step", help="rotation axis end location",default=1)
    parser.add_argument("--slice_start", help="a single slice to run center.py. Put -1 for n_slice if supplied",default=600)
    parser.add_argument("--n_slice", help="number of slices. Put -1 for slice_start if supplied",default=-1)
    parser.add_argument("--medfilt_size", help="size of median filter",default=2)
    parser.add_argument("--level", help="level of downsampling",default=0)
    parser.add_argument("--padding", help="sinogram padding", default=1000)
    parser.add_argument("--debug", help="debug messages",default=0,type=int)
    args = parser.parse_args()

    debug = args.debug
    fname = args.file_name

    fname = args.file_name

    if fname == 'auto':
        h5file = glob('*.h5')
        fname = h5file[0] 
        print ('Autofilename =' + fname)

    rot_start = int(args.rot_start)
    rot_end = int(args.rot_end)
    rot_step = int(args.rot_step)
    slice = int(args.slice_start)
    n_slice = int(args.n_slice)
    medfilt_size = int(args.medfilt_size)
    level = int(args.level)
    pad_length = int(args.padding)

    array_dims = util.read_data_adaptive(fname, shape_only=True)

    if slice == -1:
        sino_start = 200
        sino_end = array_dims[1]-200
        sino_step = int((sino_end - sino_start)) / n_slice + 1
    else:
        sino_start = slice
        sino_end = slice + 1
        sino_step = 1


    # Select the rotation center range.
    if (rot_start < 0) or (rot_end > array_dims[2]):
        rot_center = array_dims[2]/2
        rot_start = rot_center - 30
        rot_end = rot_center + 30

    if (rot_step < 0) or (rot_step > (rot_end - rot_start)):
        rot_step = 1
    center_range=[rot_start, rot_end, rot_step]
    print ("Center:", center_range)

    # Select the sinogram range to reconstruct.
    if (sino_start < 0) or (sino_start > array_dims[1]):
        sino_start = array_dims[1]/2

    folder = os.path.dirname(fname) + os.sep

    N_recon = rot_start - rot_end

    prj, flat, dark, theta = util.read_data_adaptive(fname, sino=(sino_start, sino_end, sino_step),
                                                     return_theta=True)

    # Read theta from the dataset:
    # theta = tomopy.angles(prj.shape[0])

    debug_print(debug,'## Debug: after reading data:')
    debug_print(debug,'\n** Shape of the data:'+str(np.shape(prj)))
    debug_print(debug,'** Shape of theta:'+str(np.shape(theta)))
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.normalize(prj, flat, dark)
    debug_print(debug,'\n** Flat field correction done!\n')

    debug_print(debug,'## Debug: after normalization:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.minus_log(prj)
    debug_print(debug,'\n** minus log applied!')

    debug_print(debug,'## Debug: after minus log:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
    prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
    prj[np.where(prj == np.inf)] = 0.001

    debug_print(debug,'## Debug: after cleaning bad values:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    # prj = tomopy.remove_stripe_fw(prj, 5, wname='sym16', sigma=1, pad=True)
    prj = tomopy.remove_stripe_ti(prj, 4)
    debug_print(debug,'\n** Stripe removal done!')
    debug_print(debug,'## Debug: after remove_stripe:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    if medfilt_size not in (0, None):
        prj = tomopy.median_filter(prj,size=medfilt_size)
        debug_print(debug,'\n** Median filter done!')
        debug_print(debug,'## Debug: after nedian filter:')
        debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))


    if level>0:
        prj = tomopy.downsample(prj, level=level)
        debug_print(debug,'\n** Down sampling done!\n')
        debug_print(debug,'## Debug: after down sampling:')
        debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    slice_ls = range(sino_start, sino_end, sino_step)
    prj = util.pad_sinogram(prj, pad_length)
    rot_start += pad_length
    rot_end += pad_length
    for ind, i in enumerate(slice_ls):
        debug_print(debug,'Writing center {}'.format(i))
        outpath = os.path.join(os.getcwd(), 'center', str(i))
        util.write_center(prj[:, ind:ind + 1, :], theta, dpath=outpath,
                          cen_range=[rot_start / pow(2, level), rot_end / pow(2, level),
                                     rot_step / pow(2, level)],
                          pad_length=pad_length)
    debug_print(debug,"#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
