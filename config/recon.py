#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to make a preview reconstruction.

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
    parser.add_argument("center", help="rotation center")
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
        flat_grp = '/'.join([exchange_base, 'data_white'])
        flat = dxreader.read_hdf5(fname, flat_grp)
        
        if os.path.isfile(fname):

            # Select the sinogram range to reconstruct.
            start = flat.shape[1]/2
            end = start + 2
            
            rot_center = flat.shape[2]/2
            rot_start = rot_center - 100
            rot_end = rot_center + 100

            # Read the APS raw data.
            proj, flat, dark = dxchange.read_aps_32id(fname, sino=(start, end))

            # Set data collection angles as equally spaced between 0-180 degrees.
            theta = tomopy.angles(proj.shape[0])

            # Flat-field correction of raw data.
            proj = tomopy.normalize(proj, flat, dark)

            # Use rotation center.
            rot_center = int(args.center)

            tomopy.minus_log(proj)
            rec_fname = (folder + 'recon' + os.sep) + h5_fname.split('.')[0] + '_' + args.center

            # Reconstruct object using Gridrec algorithm.
            rec = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')

            # Mask each reconstructed slice with a circle.
            rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

            # Write data as stack of TIFs.
            dxchange.write_tiff_stack(rec, fname=rec_fname)
    except:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
