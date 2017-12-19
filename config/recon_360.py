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


def debug_print(debug, text):
    if debug:
        print(text)


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("--center_folder", help="folder containing center testing images",default='center')
    parser.add_argument("--sino_start", help="slice start",default=0,type=int)
    parser.add_argument("--sino_end", help="slice end,",default=1200,type=int)
    parser.add_argument("--sino_step", help="slice step",default=1)
    parser.add_argument("--medfilt_size", help="size of median filter",default=0,type=int)
    parser.add_argument("--level", help="level of downsampling",default=0,type=int)
    parser.add_argument("--pad_lenght", help="sinogram padding",default=1000,type=int)
    parser.add_argument("--chunk_size", help="chunk size",default=50,type=int)
    parser.add_argument("--debug", help="debug messages",default=0,type=int)
    # parser.add_argument("padding", help="sinogram padding")
    args = parser.parse_args()
    
    fname = args.file_name

  
    debug = args.debug
    fname = args.file_name

    if fname == 'auto':
        h5file = glob('*.h5')
        print (h5file)
        fname = h5file[0] 
        print ('Autofilename =' + fname)

    array_dims = util.h5group_dims(fname)
    folder = os.path.dirname(fname) + os.sep

    chunk_size = args.chunk_size
    medfilt_size = args.medfilt_size
    level = args.level
    pad_length = args.pad_lenght

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

    pbar = tqdm(chunks)
    for (chunk_st, chunk_end) in pbar:

        pbar.set_description('Chunk range: ({:d}, {:d})'.format(chunk_st, chunk_end))

        prj, flat, dark, theta = util.read_data_adaptive(fname, sino=(chunk_st, chunk_end, sino_step), return_theta=True)

        # theta = tomopy.angles(prj.shape[0])

        debug_print(debug,'## Debug: after reading data:')
        debug_print(debug,'\n** Shape of the data:'+str(np.shape(prj)))
        debug_print(debug,'** Shape of theta:'+str(np.shape(theta)))
        debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.normalize(prj, flat, dark)
        debug_print(debug,'\n** Flat field correction done!')

        fov2 = int(prj.shape[2] / 2)
        axis_side = 'left' if center_pos < fov2 else 'right'
        overlap = (prj.shape[2] - center_pos) * 2 if axis_side == 'right' else center_pos * 2
        prj = util.sino_360_to_180(prj, overlap=overlap, rotation=axis_side)
        theta = theta[:prj.shape[0]]
        debug_print(debug,'\n** Sinogram converted!')

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

        prj = tomopy.remove_stripe_ti(prj,4)
        debug_print(debug,'\n** Stripe removal done!')
        debug_print(debug,'## Debug: after remove_stripe:')
        debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        if medfilt_size not in (0, None):
            prj = tomopy.median_filter(prj, size=medfilt_size)
            debug_print(debug,'\n** Median filter done!')
            print('## Debug: after nedian filter:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f' % (np.min(prj), np.max(prj)))

        if level > 0:
            prj = tomopy.downsample(prj, level=level)
            debug_print(debug,'\n** Down sampling done!\n')
            debug_print(debug,'## Debug: after down sampling:')
            debug_print(debug,'\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        recon_center = center_pos if axis_side == 'right' else (prj.shape[2] - center_pos)
        raw_shape = prj.shape
        if not pad_length == 0:
            prj = util.pad_sinogram(prj, pad_length)
        rec = tomopy.recon(prj, theta, center=recon_center+pad_length, algorithm='gridrec', filter_name='parzen')
        debug_print(debug,'\nReconstruction done!\n')

        if not pad_length == 0:
            rec = rec[:, pad_length:pad_length+raw_shape[2], pad_length:pad_length+raw_shape[2]]

        dxchange.write_tiff_stack(rec, fname=os.path.join('recon', 'recon'), start=chunk_st, dtype='float32')


if __name__ == "__main__":
    main(sys.argv[1:])
