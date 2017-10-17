#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function

import ConfigParser
import argparse
import os
import sys
import re
from os.path import expanduser

import h5py
import dxchange
import tomopy
import numpy as np

import automo.util as util


def sino_360_to_180(data, overlap=0, rotation='left'):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded.
    Parameters
    ----------
    data : ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.
    Returns
    -------
    ndarray
        Output 3D data.
    """
    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))

    lo = overlap//2
    ro = overlap - lo
    n = dx//2

    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)

    if rotation == 'left':
        out[:, :, -(dz-lo):] = data[:n, :, lo:]
        out[:, :, :-(dz-lo)] = data[n:2*n, :, ro:][:, :, ::-1]
    elif rotation == 'right':
        out[:, :, :dz-lo] = data[:n, :, :-lo]
        out[:, :, dz-lo:] = data[n:2*n, :, :-ro][:, :, ::-1]

    return out


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("rot_start", help="rotation axis start location")
    parser.add_argument("rot_end", help="rotation axis end location")
    parser.add_argument("rot_step", help="rotation axis end location")
    parser.add_argument("slice_start", help="a single slice to run center.py. Put -1 for n_slice if supplied")
    parser.add_argument("n_slice", help="number of slices. Put -1 for slice_start if supplied")
    parser.add_argument("medfilt_size", help="size of median filter")
    parser.add_argument("level", help="level of downsampling")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name
    rot_start = int(args.rot_start)
    rot_end = int(args.rot_end)
    rot_step = int(args.rot_step)
    slice = int(args.slice_start)
    n_slice = int(args.n_slice)
    medfilt_size = int(args.medfilt_size)
    level = float(args.level)

    array_dims = util.read_data_adaptive(fname, shape_only=True)

    if slice == -1:
        sino_start = 0
        sino_end = array_dims[1]-1
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

    prj, flat, dark = util.read_data_adaptive(fname, sino=(sino_start, sino_end, sino_step))

    # Read theta from the dataset:
    theta = tomopy.angles(int(prj.shape[0]//2))

    print('## Debug: after reading data:')
    print('\n** Shape of the data:'+str(np.shape(prj)))
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.normalize(prj, flat, dark)
    print('\n** Flat field correction done!\n')

    for center in range(rot_start, rot_end, rot_step):

        overlap = (prj.shape[2] - center) * 2
        prj = sino_360_to_180(prj, overlap=overlap, rotation='right')
        print('\n** Sinogram converted!')


        print('## Debug: after normalization:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.minus_log(prj)
        print('\n** minus log applied!')

        print('## Debug: after minus log:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
        prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
        prj[np.where(prj == np.inf)] = 0.001

        print('## Debug: after cleaning bad values:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.remove_stripe_ti(prj,4)
        print('\n** Stripe removal done!')
        print('## Debug: after remove_stripe:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

        prj = tomopy.median_filter(prj,size=medfilt_size)
        print('\n** Median filter done!')
        print('## Debug: after nedian filter:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))


        if level>0:
            prj = tomopy.downsample(prj, level=level)
            print('\n** Down sampling done!\n')
            print('## Debug: after down sampling:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

        rec = tomopy.recon(prj, theta, center=center, algorithm='gridrec')

        slice_ls = range(sino_start, sino_end, sino_step)
        for i in range(rec.shape[0]):
            outpath = os.path.join(os.getcwd(), 'center', str(slice_ls[i]))
            dxchange.write_tiff(rec[i], os.path.join(outpath, '{:.2f}'.format(center)))

    slice_ls = range(sino_start, sino_end, sino_step)
    center_ls = []
    for i in slice_ls:
        outpath = os.path.join(os.getcwd(), 'center', str(i))
        min_entropy_fname = util.minimum_entropy(outpath, mask_ratio=0.7, ring_removal=True)
        center_ls.append(float(re.findall('\d+\.\d+', os.path.basename(min_entropy_fname))[0]))
    if len(center_ls) == 1:
        center_pos = center_ls[0]
    else:
        center_pos = np.mean(util.most_neighbor_clustering(center_ls, 5))
    f = open('center_pos.txt', 'w')
    f.write(str(center_pos))
    f.close()
    print("#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
