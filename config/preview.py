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
import h5py
from glob import glob
import tomopy

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="existing hdf5 file name",default='auto')
    parser.add_argument("--proj_start", help="preview projection start; for full rec enter -1", default='auto')
    parser.add_argument("--proj_end", help="preview projection end; for full rec enter -1", default='auto')
    parser.add_argument("--proj_step", help="preview projection step; for full rec enter -1", default='auto')
    parser.add_argument("--slice_start", help="preview slice start; for full rec enter -1", default='auto')
    parser.add_argument("--slice_end", help="preview slice end; for full rec enter -1", default='auto')
    parser.add_argument("--slice_step", help="preview slice step; for full rec enter -1", default='auto')
    args = parser.parse_args()


    fname = args.file_name

    if fname == 'auto':
        h5file = glob('*.h5')
        fname = h5file[0] 
        print ('Autofilename =' + fname)
        
    folder = './'

    try:
        proj_st = int(args.proj_start)
        proj_end = int(args.proj_end)
        proj_step = int(args.proj_step)
        slice_st = int(args.slice_start)
        slice_end = int(args.slice_end)
        slice_step = int(args.slice_step)
    except:
        h5 = h5py.File(fname)
        dset = h5['exchange/data'] #this should be adaptative
        proj_st = 0
        proj_end = 1
        proj_step = 1
        slice_st = int(dset.shape[1] / 2)
        slice_end = slice_st + 1
        slice_step = 1

    if os.path.isfile(fname):

        # h5 = h5py.File(fname)
        # dset = h5['exchange/data'] #this should be adaptative

        # Read the APS raw data projections.
        proj, flat, dark, _ = util.read_data_adaptive(fname, proj=(proj_st, proj_end, proj_step))
        print("Proj Preview: ", proj.shape)
        
        proj_norm = tomopy.normalize(proj, flat, dark)
        proj_norm = tomopy.minus_log(proj_norm)
        proj_norm = tomopy.misc.corr.remove_neg(proj_norm, val=0.001)
        proj_norm = tomopy.misc.corr.remove_nan(proj_norm, val=0.001)
        proj_norm[np.where(proj_norm == np.inf)] = 0.001
        #proj_norm = util.preprocess(proj_norm) #awkward
        #proj_norm = proj_norm.astype('int16') (need a better cast)

        proj_fname = (folder + 'preview' + os.sep + 'proj')
        proj_norm_fname = (folder + 'preview' + os.sep + 'proj_norm')
        print("Proj folder: ", proj_fname)
        
        flat = flat.astype('float16')
        dxchange.write_tiff(flat.mean(axis=0), fname=(folder + 'preview' + os.sep + 'flat'), overwrite=True)

        sino, flat, dark, _ = util.read_data_adaptive(fname, sino=(slice_st, slice_end, slice_step))
        print("Sino Preview: ", sino.shape)

        sino_fname = (folder + 'preview' + os.sep + 'sino')
        sino = np.swapaxes(sino, 0, 1)
        print("Proj folder: ", proj_fname)

        dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)
        dxchange.write_tiff_stack(proj_norm, fname=proj_norm_fname, axis=0, digit=5, start=0, overwrite=True)
        dxchange.write_tiff_stack(sino, fname=sino_fname, axis=0, digit=5, start=slice_st, overwrite=True)
        print("#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
