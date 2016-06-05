#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct tomography data with different center.

"""

from __future__ import print_function

import ConfigParser
import argparse
import os
import sys
from os.path import expanduser

import dxchange
import tomopy

import automo.util as util


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("rot_start", help="rotation axis start location")
    parser.add_argument("rot_end", help="rotation axis end location")
    parser.add_argument("rot_step", help="rotation axis end location")
    parser.add_argument("slice", help="slice to run center.py")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name

    array_dims = util.h5group_dims(fname)

    # Select the rotation center range.
    rot_start = int(args.rot_start)
    rot_end = int(args.rot_end)    
    if (rot_start < 0) or (rot_end > array_dims[2]):
        rot_center = array_dims[2]/2
        rot_start = rot_center - 30
        rot_end = rot_center + 30

    rot_step = args.rot_step
    if (rot_step < 0) or (rot_step > (rot_end - rot_start)):
        rot_step = 1
    center_range=[rot_start, rot_end, rot_step]
    print ("Center:", center_range)

    # Select the sinogram range to reconstruct.
    sino_start = int(args.slice)
    if (sino_start < 0) or (sino_start > array_dims[1]):
        sino_start = array_dims[1]/2

    sino_end = sino_start + 2
    sino = [sino_start, sino_end]
    print ("Sino: ", sino)

    folder = os.path.dirname(fname) + os.sep
    try:        
        if os.path.isfile(fname):
            # Read the APS raw data.
            proj, flat, dark = dxchange.read_aps_32id(fname, sino=sino)
            
            # Set data collection angles as equally spaced between 0-180 degrees.
            theta = tomopy.angles(proj.shape[0])

            # Flat-field correction of raw data.
            proj = tomopy.normalize(proj, flat[15:20], dark[8:10])
            
            tomopy.minus_log(proj)

            rec_fname = (folder + 'center' + os.sep)
            print("Rec folder: ", rec_fname)
            rec = tomopy.write_center(proj, theta, dpath=rec_fname, cen_range=[rot_start, rot_end, rot_step], ind=0, mask=True)
            print("#################################")
    except:
        print (folder, 'does not contain the expected file hdf5 file')
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
