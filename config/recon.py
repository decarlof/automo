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
from os.path import expanduser

import dxchange
import tomopy

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

    fname = args.file_name

    array_dims = util.h5group_dims(fname)

    chunk_size = int(args.chunk_size)

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
        chunk_end += chunk_size
    chunks.append((chunk_st, sino_end))


    sino_range = range(1, array_dims[1])
    if not ((sino_start < sino_end) and sino_start in sino_range and sino_end in sino_range):
        sino_start = 1
        sino_end = array_dims[1]

    sino_step_range = range(0, array_dims[1]-1)
    if not sino_step in sino_step_range:
        sino_step =50
    
    sino = [sino_start, sino_end, sino_step]
    folder = os.path.dirname(fname) + os.sep

    try:

        if os.path.isfile(fname):

            sino_st /= pow(2, level)
            sino_end /= pow(2, level)
            Center /= pow(2, level)

            chunks = []
            chunk_st = sino_st
            chunk_end = chunk_st + chunk_size

            while chunk_end < sino_end:
                chunks.append((chunk_st, chunk_end))
                chunk_st = chunk_end
                chunk_end += chunk_size
            chunks.append((chunk_st, sino_end))

            for (chunk_st, chunk_end) in chunks:

                print('Chunk range: ({:d}, {:d})'.format(chunk_st, chunk_end))

                try:
                    prj, flat, dark, theta = dxchange.read_aps_32id(file_name, sino=(chunk_st, chunk_end))
                    print(prj.shape, flat.shape, dark.shape)
                except:
                    try:
                        prj, flat, dark = dxchange.read_aps_32id(file_name, sino=(chunk_st, chunk_end))
                        print(prj.shape, flat.shape, dark.shape)
                        f = h5py.File(file_name, "r"); dset_theta = f["/exchange/theta"]; theta = dset_theta[...]; theta = theta*np.pi/180
                    except:
                        f = h5py.File(file_name, "r")
                        prj = f['exchange/data'][:, chunk_st:chunk_end, :].astype('float32')
                        flat = f['exchange/data_white'][:, chunk_st:chunk_end, :].astype('float32')
                        dark = f['exchange/data_dark'][:, chunk_st:chunk_end, :].astype('float32')
                        theta = f['exchange/theta'].value.astype('float32')
                        theta = theta*np.pi/180


                if debug:
                    print('## Debug: after reading data:')
                    print('\n** Shape of the data:'+str(np.shape(prj)))
                    print('** Shape of theta:'+str(np.shape(theta)))
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                prj = tomopy.normalize(prj, flat, dark)
                print('\n** Flat field correction done!')

                print(prj)

                if debug:
                    print('## Debug: after normalization:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                prj = tomopy.minus_log(prj)
                print('\n** minus log applied!')

                if debug:
                    print('## Debug: after minus log:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
                prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
                prj[np.where(prj == np.inf)] = 0.001

                if debug:
                    print('## Debug: after cleaning bad values:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                prj = tomopy.remove_stripe_ti(prj,4)
                print('\n** Stripe removal done!')
                if debug:
                    print('## Debug: after remove_stripe:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                prj = tomopy.median_filter(prj,size=medfilt_size)
                print('\n** Median filter done!')
                if debug:
                    print('## Debug: after nedian filter:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))


                if level>0:
                    prj = tomopy.downsample(prj, level=level)
                    print('\n** Down sampling done!\n')
                if debug:
                    print('## Debug: after down sampling:')
                    print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

                rec = tomopy.recon(prj, theta, center=Center, algorithm='gridrec', filter_name='parzen')
                print('\nReconstruction done!\n')

                dxchange.write_tiff_stack(rec, fname=output_name, start=chunk_st, dtype='float32')



    except:
        print(folder, 'does not contain the expected file hdf5 file')
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
