#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to generate a series of preview projections.

"""

from __future__ import print_function
import os
import sys
import dxchange
import argparse
import ConfigParser
from os.path import expanduser
import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("slice_start", help="recon slice start; for full rec enter -1")
    parser.add_argument("slice_end", help="recon slice end; for full rec enter -1")
    parser.add_argument("slice_step", help="recon slice step; for full rec enter -1")
    parser.add_argument("rot_center", help="rotation center; for auto center enter -1")
    #parser.add_argument("save_dir", help="relative save directory")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name

    array_dims = util.h5group_dims(fname)

    folder = os.path.dirname(fname) + os.sep
 
    try: 
        if os.path.isfile(fname):

            # Read the APS raw data projections.
            proj, flat, dark = dxchange.read_aps_32id(fname, proj=(0, array_dims[0], 20))
            print("Proj Preview: ", proj.shape)        

            proj_fname = (folder + 'preview' + os.sep + 'data')
            print("Proj folder: ", proj_fname)        

            dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)          
            print("#################################")

    except:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
