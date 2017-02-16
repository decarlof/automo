#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to generate a series of preview projections.

"""

from __future__ import print_function

import ConfigParser
import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("proj_start", help="preview projection start; for full rec enter -1")
    parser.add_argument("proj_end", help="preview projection end; for full rec enter -1")
    parser.add_argument("proj_step", help="preview projection step; for full rec enter -1")
    parser.add_argument("slice_start", help="preview slice start; for full rec enter -1")
    parser.add_argument("slice_end", help="preview slice end; for full rec enter -1")
    parser.add_argument("slice_step", help="preview slice step; for full rec enter -1")
    # parser.add_argument("rot_center", help="rotation center; for auto center enter -1")
    #parser.add_argument("save_dir", help="relative save directory")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name
    proj_st = int(args.proj_start)
    proj_end = int(args.proj_end)
    proj_step = int(args.proj_step)
    slice_st = int(args.slice_start)
    slice_end = int(args.slice_end)
    slice_step = int(args.slice_step)

    # rot_center = args.rot_center

    #folder = os.path.dirname(fname) + os.sep
    folder = './'

    try: 
        if os.path.isfile(fname):

            # Read the APS raw data projections.
            proj, flat, dark = dxchange.read_aps_32id(fname, proj=(proj_st, proj_end, proj_step))
            print("Proj Preview: ", proj.shape)        

            proj_fname = (folder + 'preview' + os.sep + 'proj')
            print("Proj folder: ", proj_fname)        

            sino, flat, dark = dxchange.read_aps_32id(fname, sino=(slice_st, slice_end, slice_step))
            print("Proj Preview: ", proj.shape)

            sino_fname = (folder + 'preview' + os.sep + 'sino')
            print("Proj folder: ", proj_fname)

            dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)          
            dxchange.write_tiff_stack(sino, fname=sino_fname, axis=0, digit=5, start=0, overwrite=True)
            print("#################################")

    except:
        warnings.warn('Booooooom')


if __name__ == "__main__":
    main(sys.argv[1:])
