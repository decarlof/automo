#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function
import os
import sys
import tomopy
import dxchange
import automo
import automo.util as util
import argparse
import ConfigParser
from os.path import expanduser
import dxchange.reader as dxreader


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="existing data folder")
    parser.add_argument("rot_start", help="rotation axis start location")
    parser.add_argument("rot_end", help="rotation axis end location")
    parser.add_argument("slice", help="slice to run center.py")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    default_h5_fname = cf.get('settings', 'default_h5_fname')

    # will add the trailing slash if it's not already there.
    folder = os.path.normpath(automo._clean_folder_name(args.folder)) + os.sep 
    fname = folder + default_h5_fname

    array_dims = util.h5group_dims(fname)
    print (array_dims)

    # Select the rotation center range.
    rot_start = int(args.rot_start)
    rot_end = int(args.rot_end)    
    if (rot_start < 0) or (rot_end > array_dims[2]):
        rot_center = array_dims[2]/2
        rot_start = rot_center - 150
        rot_end = rot_center + 150

    center_range=[rot_start, rot_end, 5]
    print ("Center:", center_range)

    # Select the sinogram range to reconstruct.
    sino_start = int(args.slice)
    if (sino_start < 0) or (sino_start > array_dims[1]):
        sino_start = array_dims[1]/2

    sino_end = sino_start + 2
    sino = [sino_start, sino_end]
    print ("Sino:", sino)
        
    exchange_base = "exchange"
    flat_grp = '/'.join([exchange_base, 'data_white'])

    try: 

        flat = dxreader.read_hdf5(fname, flat_grp)
        
        if os.path.isfile(fname):
            # Read the APS raw data.
            proj, flat, dark = dxchange.read_aps_32id(fname, sino=sino)
            print (proj.shape, flat.shape, dark.shape)
            
            # Set data collection angles as equally spaced between 0-180 degrees.
            theta = tomopy.angles(proj.shape[0])

            # Flat-field correction of raw data.
            proj = tomopy.normalize(proj, flat, dark)
            print (proj.shape)
            
            tomopy.minus_log(proj)

            rec_fname = (folder + 'center' + os.sep).split('.')[0]
            #rec = tomopy.write_center(proj, theta, dpath=rec_fname, center_range=[rot_start, rot_end, 5], ind=0, mask=True)
    except:
        print (folder, 'does not contain the expected file:', default_h5_fname)
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
