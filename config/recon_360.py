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
import shutil
from os.path import expanduser

import dxchange
import tomopy
import numpy as np
import h5py
from glob import glob
from tqdm import tqdm

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
    parser.add_argument("file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("center_folder", help="folder containing center testing images")
    parser.add_argument("sino_start", help="slice start")
    parser.add_argument("sino_end", help="slice end")
    parser.add_argument("sino_step", help="slice step")
    parser.add_argument("medfilt_size", help="size of median filter")
    parser.add_argument("level", help="level of downsampling")
    parser.add_argument("chunk_size", help="chunk size")
    args = parser.parse_args()

    fname = args.file_name

    if fname == 'auto':
        h5file = glob('*.h5')
        fname = h5file[0] 
        print ('Autofilename =' + 

    array_dims = util.h5group_dims(fname)
    folder = os.path.dirname(fname) + os.sep

    chunk_size = int(args.chunk_size)
    medfilt_size = int(args.medfilt_size)
    level = int(args.level)

    pad_length = 1000

    # find center if not given
    if os.path.exists('center_pos.txt'):
        f = open('center_pos.txt')
        center_pos = f.readline()
        center_pos = float(center_pos)
        f.close()
    else:
        slice_ls = os.listdir('center')
        for ind, i in enumerate(slice_ls):
            if not os.path.isfile(i):
                slice_ls[ind] = int(i)
        center_ls = []
        for i in slice_ls:
            outpath = os.path.join(os.getcwd(), 'center', str(i))
            center_pos = util.minimum_entropy(outpath, mask_ratio=0.7, ring_removal=True)
            center_ls.append(center_pos)
        if len(center_ls) == 1:
            center_pos = center_ls[0]
        else:
            center_pos = np.mean(util.most_neighbor_clustering(center_ls, 5))
        f = open('center_pos.txt', 'w')
        f.write(str(center_pos))
        f.close()


    # perform reconstruction
    # try:

    print("Data: ", array_dims)
    # Select the sinogram range to reconstruct.
    sino_start = int(args.sino_start)
    sino_end = int(args.sino_end)
    sino_step = int(args.sino_step)

    chunks = []
    chunk_st = sino_start
    chunk_end = chunk_st + chunk_size * sino_step

    while chunk_end < sino_end:
        chunks.append((chunk_st, chunk_end))
        chunk_st = chunk_end
        chunk_end += chunk_size * sino_step
    chunks.append((chunk_st, sino_end))

    for (chunk_st, chunk_end) in chunks:

        print('Chunk range: ({:d}, {:d})'.format(chunk_st, chunk_end))

        prj, flat, dark, theta = util.read_data_adaptive(file_name, sino=(chunk_st, chunk_end, sino_step), return_theta=True)

        # theta = tomopy.angles(prj.shape[0])

        print('## Debug: after reading data:')
        print('\n** Shape of the data:'+str(np.shape(prj)))
        print('** Shape of theta:'+str(np.shape(theta)))
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.normalize(prj, flat, dark)
        print('\n** Flat field correction done!')

        fov2 = int(prj.shape[2] / 2)
        axis_side = 'left' if center_pos < fov2 else 'right'
        overlap = (prj.shape[2] - center_pos) * 2 if axis_side == 'right' else center_pos * 2
        prj = sino_360_to_180(prj, overlap=overlap, rotation=axis_side)
        theta = theta[:prj.shape[0]]
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
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        if medfilt_size not in (0, None):
            prj = tomopy.median_filter(prj, size=medfilt_size)
            print('\n** Median filter done!')
            print('## Debug: after nedian filter:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

        if level > 0:
            prj = tomopy.downsample(prj, level=level)
            print('\n** Down sampling done!\n')
            print('## Debug: after down sampling:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        recon_center = center_pos if axis_side == 'right' else (prj.shape[2] - center_pos)
        raw_shape = prj.shape
        if not pad_length == 0:
            prj = util.pad_sinogram(prj, pad_length)
        rec = tomopy.recon(prj, theta, center=recon_center+pad_length, algorithm='gridrec', filter_name='parzen')
        print('\nReconstruction done!\n')

        if not pad_length == 0:
            rec = rec[:, pad_length:pad_length+raw_shape[2], pad_length:pad_length+raw_shape[2]]

        dxchange.write_tiff_stack(rec, fname=os.path.join('recon', 'recon'), start=chunk_st, dtype='float32')


if __name__ == "__main__":
    main(sys.argv[1:])
