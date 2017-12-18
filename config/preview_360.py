#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo example to generage sample previews from 360 acquisitions.
"""

from __future__ import print_function

import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob
import h5py

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name",default='auto')
    args = parser.parse_args()


    fname = args.file_name

    if fname == 'auto':
        h5file = glob('*.h5')
        fname = h5file[0] 
        print ('Autofilename =' + h5file)


    folder = './'

#    # write_stand-alone scripts
#    for script in ['hdf5_frame_writer_360_standalone.py', 'center_360_standalone.py', 'recon_360_standalone.py']:
#        if os.path.exists(os.path.join(home, '.automo', script)):
#            shutil.copyfile(os.path.join(home, '.automo', script), script)

    if os.path.isfile(fname):

        h5 = h5py.File(fname)
        dset = h5['exchange/data']
        proj_st = 0
        proj_end = int(dset.shape[0]/2) + 1
        proj_step = int(dset.shape[0]/2)
        slice_st = int(dset.shape[1]/2)
        slice_end = slice_st+1
        slice_step = 1


        # Read the APS raw data projections.
        proj, flat, dark, _ = util.read_data_adaptive(fname, proj=(proj_st, proj_end, proj_step))
        print("Proj Preview: ", proj.shape)

        proj_fname = (folder + 'preview' + os.sep + 'proj')
        print("Proj folder: ", proj_fname)

        sino, flat, dark, _ = util.read_data_adaptive(fname, sino=(slice_st, slice_end, slice_step))
        print("Sino Preview: ", sino.shape)
    
        sino_fname = (folder + 'preview' + os.sep + 'sino')
        sino = np.swapaxes(sino, 0, 1)
        print("Proj folder: ", proj_fname)

        dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)
        dxchange.write_tiff_stack(sino, fname=sino_fname, axis=0, digit=5, start=slice_st, overwrite=True)
        proj_flip = np.fliplr(proj[1,:,:])
        proj_fname = (folder + 'preview' + os.sep + 'proj_00001_flip')
        dxchange.write_tiff(proj_flip, fname=proj_fname, overwrite=True)
        print("#################################")



if __name__ == "__main__":
    main(sys.argv[1:])
