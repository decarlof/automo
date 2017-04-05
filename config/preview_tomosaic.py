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
import numpy as np
import tomopy
import re

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
    parser.add_argument("write_aux", help="whether to write flat/dark fields or not")
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
    write_aux = True if (args.write_aux == 'True') else False

    # rot_center = args.rot_center

    #folder = os.path.dirname(fname) + os.sep
    folder = './'

    if os.path.isfile(fname):

        regex = re.compile(r'(.+)_y(\d+)_x(\d+).h5')
        iy, ix = regex.search(fname).group(2, 3)

        # Read the APS raw data projections.
        proj, flat, dark = dxchange.read_aps_32id(fname, proj=(proj_st, proj_end, proj_step))
        print("Proj shape: ", proj.shape)

        proj_fname = (folder + 'preview' + os.sep + 'y{:s}_x{:s}'.format(iy, ix) + os.sep + 'proj' + os.sep + 'proj')
        print("Proj folder: ", proj_fname)

        dxchange.write_tiff_stack(proj, fname=proj_fname, axis=0, digit=5, start=0, overwrite=True)

        if write_aux:

            flat_fname = (folder + 'preview' + os.sep + 'y{:s}_x{:s}'.format(iy, ix) + os.sep + 'flat' + os.sep + 'flat')
            dxchange.write_tiff_stack(flat, fname=flat_fname, axis=0, digit=5, start=0, overwrite=True)

            dark_fname = (folder + 'preview' + os.sep + 'y{:s}_x{:s}'.format(iy, ix) + os.sep + 'dark' + os.sep + 'dark')
            dxchange.write_tiff_stack(dark, fname=dark_fname, axis=0, digit=5, start=0, overwrite=True)

        sino, flat, dark = dxchange.read_aps_32id(fname, sino=(slice_st, slice_end, slice_step))
        print("Sino preview: ", sino.shape)

        sino_fname = (folder + 'preview' + os.sep + 'y{:s}_x{:s}'.format(iy, ix) + os.sep + 'sino' + os.sep + 'sino')
        sino = np.swapaxes(sino, 0, 1)

        dxchange.write_tiff_stack(sino, fname=sino_fname, axis=0, digit=5, start=0, overwrite=True)

        # write projecions for rough manual registration
        proj, flat, dark = dxchange.read_aps_32id(fname, proj=(0, 1))
        proj = tomopy.normalize(proj, flat, dark)
        dxchange.write_tiff(proj, os.path.join('manual_regist', 'y{:s}_x{:s}'.format(iy, ix)))

        print("#################################")


if __name__ == "__main__":
    main(sys.argv[1:])