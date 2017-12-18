#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo example to generage sample previews.
"""

from __future__ import print_function

import six.moves.configparser as ConfigParser
import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name",default='auto')
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name

    if fname = 'auto':
        h5file = glob.glob('*.h5')
        fname = h5file[0] 
        print ('Autofilename =' + h5file)


        
    folder = './'

    if os.path.isfile(fname):

        h5 = File(fname)
        dset = h5['exchange/data'] #this should be adaptative
        proj_st = 0
        proj_end = proj_st + 1
        proj_step = 1
        slice_st = 600
        slice_end = slice_st+1
        slice_step = 1


        # Read the APS raw data projections.
        proj, flat, dark, _ = util.read_data_adaptive(fname, proj=(proj_st, proj_end, proj_step))
        print("Proj Preview: ", proj.shape)

        proj_fname = (folder + 'preview' + os.sep + 'proj')
        print("Proj folder: ", proj_fname)

        dxchange.write_tiff(flat.mean(axis=0), fname=(folder + 'preview' + os.sep + 'flat'), overwrite=True)

        sino, flat, dark, _ = util.read_data_adaptive(fname, sino=(slice_st, slice_end, slice_step))
        print("Sino Preview: ", sino.shape)

        sino_fname = (folder + 'preview' + os.sep + 'sino')
        sino = np.swapaxes(sino, 0, 1)
        print("Proj folder: ", proj_fname)

        dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)
        dxchange.write_tiff_stack(sino, fname=sino_fname, axis=0, digit=5, start=slice_st, overwrite=True)
        print("#################################")


if __name__ == "__main__":
    main(sys.argv[1:])
