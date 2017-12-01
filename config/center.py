#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function

import six.moves.configparser as ConfigParser
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
try:
    import xlearn
except:
    warnings.warn('Cannot import package xlearn.')

import automo.util as util


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
    # parser.add_argument("padding", help="sinogram padding")
    args = parser.parse_args()

    search_method = 'dnn'
    # pad_length = int(args.padding)
    pad_length = 1000

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

    print('## Debug: after reading data:')
    print('\n** Shape of the data:'+str(np.shape(prj)))
    print('** Shape of theta:'+str(np.shape(theta)))
    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

    prj = tomopy.normalize(prj, flat, dark)
    print('\n** Flat field correction done!\n')

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

    # prj = tomopy.remove_stripe_fw(prj, 5, wname='sym16', sigma=1, pad=True)
    prj = tomopy.remove_stripe_ti(prj, 4)
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

    slice_ls = range(sino_start, sino_end, sino_step)
    center_ls = []
    if search_method == 'entropy' and pad_length > 0:
        prj = util.pad_sinogram(prj, pad_length)
        rot_start += pad_length
        rot_end += pad_length
    for ind, i in enumerate(slice_ls):
        outpath = os.path.join(os.getcwd(), 'center', str(i))
        if search_method == 'entropy':
            util.write_center(prj[:, ind:ind+1, :], theta, dpath=outpath,
                              cen_range=[rot_start/pow(2,level), rot_end/pow(2, level),
                                         rot_step/pow(2, level)],
                              pad_length=pad_length)
            min_entropy_fname = util.minimum_entropy(outpath, mask_ratio=0.4, ring_removal=False)
            center_pos = float(re.findall('\d+\.\d+', os.path.basename(min_entropy_fname))[0])
        elif search_method == 'vo':
            mid = prj.shape[2] / 2 / pow(2,level)
            smin = (rot_start/pow(2,level) - mid) * 2
            smax = (rot_end/pow(2,level) - mid) * 2
            center_pos = util.find_center_vo(prj, smin=smin, smax=smax, step=rot_step)
        elif search_method == 'dnn':
            center_pos = util.find_center_dnn(prj[:, ind:ind+1, :], theta,
                                              search_range=[rot_start/pow(2,level), rot_end/pow(2, level)],
                                              search_step=rot_step/pow(2, level),
                                              outpath=outpath,
                                              pad_length=pad_length)
        center_ls.append(center_pos)
        print('Center for slice: {}'.format(center_pos))
    if len(center_ls) == 1:
        center_pos = center_ls[0]
    else:
        center_pos = np.mean(util.most_neighbor_clustering(center_ls, 5), dtype='float')
    f = open('center_pos.txt', 'w')
    f.write(str(center_pos))
    f.close()
    print("#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
