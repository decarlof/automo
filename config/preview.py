#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to generate a series of preview projections.

"""

from __future__ import print_function
import os
import sys
import tomopy
import dxchange
import automo
import argparse
import ConfigParser
from os.path import expanduser
import dxchange.reader as dxreader


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="existing folder")
    args = parser.parse_args()
    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    h5_fname = cf.get('settings', 'h5_fname')
 
    try: 
        # will add the trailing slash if it's not already there.
        folder = os.path.normpath(automo.clean_folder_name(args.folder)) + os.sep 
        fname = folder + h5_fname

        exchange_base = "exchange"
        tomo_grp = '/'.join([exchange_base, 'data'])
        proj, flat, dark = dxchange.read_aps_32id(fname)
        tomo = dxreader.read_hdf5(fname, tomo_grp, slc=((0,(proj.shape)[0], 50), None))

        proj_fname = (folder + 'preview' + os.sep) + h5_fname.split('.')[0]
        dxchange.write_tiff_stack(tomo, fname=proj_fname, axis=0, digit=5, start=0, overwrite=False)          

    except:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
