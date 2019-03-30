#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function

import argparse
import os
import sys
import shutil
import re
from os.path import expanduser
import warnings

import h5py
import dxchange
from scipy.ndimage import rotate
import numpy as np
from glob import glob
from tqdm import tqdm
from mpi4py import MPI

import automo.util as util


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

def debug_print(debug, text):
    if debug:
        print(text)

def allocate_mpi_subsets(n_task, size, task_list=None):

    if task_list is None:
        task_list = range(n_task)
    sets = []
    for i in range(size):
        sets.append(task_list[i:n_task:size])
    return sets

def write_data(dset_f, dset_o, angle, chunk_size):

    task_list = range(0, dset_o.shape[0], chunk_size)
    sets = allocate_mpi_subsets(len(task_list), size, task_list)
    for i in sets[rank]:
        print('Block starting {:d}'.format(i))
        end = min([i+chunk_size, dset_o.shape[0]])
        dset = dset_o[i:end, :, :]
        dset = rotate(dset, angle, axes=(1, 2), reshape=False, mode='nearest')
        dset = dset.astype(np.uint16)
        dset_f[i:end, :, :] = dset
    comm.Barrier()

def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("--tilt", help="amount of tilt in degree",default='auto')
    parser.add_argument("--chunk_size", help="number of projection for batch processing",default=50)
    args = parser.parse_args()

    fname = args.file_name
    new_fname = os.path.splitext(fname)[0] + '_tilt_fixed.h5'

    tilt = args.tilt
    chunk_size = args.chunk_szie

    if fname == 'auto':
        h5file = glob('*.h5')
        fname = h5file[0]
        print ('Autofilename =' + fname)

    try:
        tilt = float(tilt)
    except:
        f = open('tilt.txt')
        tilt = float(f.readlines()[0])
        f.close()

    if rank == 0:
        o = h5py.File(fname, 'r')
        f = h5py.File(new_fname)
    comm.Barrier()
    if rank != 0:
        o = h5py.File(fname, 'r')
        f = h5py.File(new_fname)

    o_data = o['exchange/data']
    o_flat = o['exchange/data_white']
    o_dark = o['exchange/data_dark']
    o_theta = o['exchange/theta']

    grp = f.create_group('exchange')
    f_data = grp.create_dataset('data', o_data.shape, dtype=np.uint16)
    f_flat = grp.create_dataset('data_white', o_flat.shape, dtype=np.uint16)
    f_dark = grp.create_dataset('data_dark', o_data.shape, dtype=np.uint16)
    f_theta = grp.create_dataset('theta', o_theta.shape, dtype=np.uint16)
    comm.Barrier()

    write_data(f_data, o_data, tilt, chunk_size)
    write_data(f_flat, o_flat, tilt, chunk_size)
    write_data(f_dark, o_dark, tilt, chunk_size)

    f_theta[...] = o_data[...]

    f.close()
    o.close()

    os.makedirs('old_h5')
    shutil.move(fname, os.path.join('old_h5', fname))


if __name__ == "__main__":
    main(sys.argv[1:])
