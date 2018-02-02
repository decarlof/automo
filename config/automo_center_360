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

import h5py
import dxchange
import tomopy
from glob import glob
import numpy as np
from tqdm import tqdm

import automo.util as util


def debug_print(debug, text):
    if debug:
        print(text)


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("--rot_start", help="rotation axis start location",default = 800)
    parser.add_argument("--rot_end", help="rotation axis end location", default = 1100)
    parser.add_argument("--rot_step", help="rotation axis end location",default=1)
    parser.add_argument("--slice_start", help="a single slice to run center.py. Put -1 for n_slice if supplied",default=600)
    parser.add_argument("--n_slice", help="number of slices. Put -1 for slice_start if supplied",default = -1)
    parser.add_argument("--medfilt_size", help="size of median filter", default=1)
    parser.add_argument("--level", help="level of                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            downsampling", default=0)
    parser.add_argument("--debug", help="debug messages",default=0,type=int)
    args = parser.parse_args()
                                                                                                                                                                                                                                                            

    fname = args.file_name                                                                                                                                                                                                                                      
    debug = args.debug
    
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
    level = float(args.level)

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

    prj, flat, dark, theta = util.read_data_adaptive(fname, sino=(sino_start, sino_end, sino_step), return_theta=True)

    max_size = max([2 * rot_end + 2, (prj.shape[2] - rot_start) * 2 + 2])


    # Read theta from the dataset:
    # theta = tomopy.angles(int(prj.shape[0]//2))

    debug_print(debug,'## Debug: after reading data:')
    debug_print(debug,'\n** Shape of the data:'+str(np.shape(prj)))
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.normalize(prj, flat, dark)
    debug_print(debug,'\n** Flat field correction done!\n')

    debug_print(debug,'## Debug: after normalization:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    prj = tomopy.minus_log(prj)
    debug_print(debug,'\n** minus log applied!')

    debug_print(debug,'## Debug: after minus log:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
    prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
    prj[np.where(prj == np.inf)] = 0.001

    debug_print(debug,'## Debug: after cleaning bad values:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    prj = tomopy.remove_stripe_ti(prj, 4)
    debug_print(debug,'\n** Stripe removal done!')
    debug_print(debug,'## Debug: after remove_stripe:')
    debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    if medfilt_size not in (0, None):
        prj = tomopy.median_filter(prj,size=medfilt_size)
        debug_print(debug,'\n** Median filter done!')
        debug_print(debug,'## Debug: after nedian filter:')
        debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

    fov2 = int(prj.shape[2] / 2)

    pabs = tqdm(range(rot_start, rot_end, rot_step))
    for center in pabs:
        axis_side = 'left' if center < fov2 else 'right'
        overlap = (prj.shape[2] - center) * 2 if axis_side == 'right' else center * 2
        prj0 = util.sino_360_to_180(prj, overlap=overlap, rotation=axis_side)
        theta0 = theta[:prj0.shape[0]]

        if level>0:
            prj = tomopy.downsample(prj, level=level)
            debug_print(debug,'\n** Down sampling done!\n')
            debug_print(debug,'## Debug: after down sampling:')
            debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

        recon_center = center if axis_side == 'right' else (prj.shape[2] - center)
        rec = tomopy.recon(prj0, theta0, center=recon_center, algorithm='gridrec')

        out = np.zeros([rec.shape[0], max_size, max_size])
        out[:, :rec.shape[1], :rec.shape[2]] = rec

        slice_ls = range(sino_start, sino_end, sino_step)
        for i in range(out.shape[0]):
            outpath = os.path.join(os.getcwd(), 'center', str(slice_ls[i]))
            dxchange.write_tiff(out[i], os.path.join(outpath, '{:.2f}'.format(center)), dtype='float32')

    debug_print(debug,"#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
