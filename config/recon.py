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
import shutil
from os.path import expanduser

import dxchange
import tomopy
import numpy as np
import h5py

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("center_folder", help="folder containing center testing images")
    parser.add_argument("sino_start", help="slice start")
    parser.add_argument("sino_end", help="slice end")
    parser.add_argument("sino_step", help="slice step")
    parser.add_argument("medfilt_size", help="size of median filter")
    parser.add_argument("level", help="level of downsampling")
    parser.add_argument("chunk_size", help="chunk size")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    file_name = args.file_name
    array_dims = util.h5group_dims(file_name)
    folder = os.path.dirname(file_name) + os.sep

    chunk_size = int(args.chunk_size)
    medfilt_size = int(args.medfilt_size)
    level = int(args.level)

    # write_stand-alone recon script
    if os.path.exists(os.path.join(home, '.automo', 'recon_standalone.py')):
        shutil.copyfile(os.path.join(home, '.automo', 'recon_standalone.py'), 'recon.py')

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
            min_entropy_fname = util.minimum_entropy(outpath, mask_ratio=0.7, ring_removal=True)
            center_ls.append(float(re.findall('\d+\.\d+', os.path.basename(min_entropy_fname))[0]))
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

        prj, flat, dark = util.read_data_adaptive(file_name, sino=(sino_start, sino_end, sino_step))

        theta = tomopy.angles(prj.shape[0])

        print('## Debug: after reading data:')
        print('\n** Shape of the data:'+str(np.shape(prj)))
        print('** Shape of theta:'+str(np.shape(theta)))
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.normalize(prj, flat, dark)
        print('\n** Flat field correction done!')

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

        prj = tomopy.median_filter(prj,size=medfilt_size)
        print('\n** Median filter done!')
        print('## Debug: after nedian filter:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.downsample(prj, level=level)
        print('\n** Down sampling done!\n')
        print('## Debug: after down sampling:')
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        rec = tomopy.recon(prj, theta, center=center_pos, algorithm='gridrec', filter_name='parzen')
        print('\nReconstruction done!\n')

        dxchange.write_tiff_stack(rec, fname=os.path.join('recon', 'recon'), start=chunk_st, dtype='float32')



if __name__ == "__main__":
    main(sys.argv[1:])
