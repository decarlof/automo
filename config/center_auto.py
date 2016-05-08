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
import dxchange


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("rot_center", help="rotation axis starting location")
    parser.add_argument("slice", help="slice to run center_auto.py")
    parser.add_argument("dummy_02", help="not used, enter -1")
    parser.add_argument("dummy_03", help="not used, enter -1")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name

    array_dims = util.h5group_dims(fname)

    # Select the rotation center range.
    rot_center = args.rot_center
    if (rot_center < 0) or (rot_center > array_dims[2]):
        rot_center = array_dims[2]/2

    # Select the sinogram range to reconstruct.
    sino_start = int(args.slice)
    if (sino_start < 0) or (sino_start > array_dims[1]):
        sino_start = array_dims[1]/2

    sino_end = sino_start + 2
    sino = [sino_start, sino_end]
    print ("Sino:", sino)

    print (fname)
    folder = os.path.dirname(fname) + os.sep

    try:        
        if os.path.isfile(fname):
            # Read the APS raw data.
            proj, flat, dark = dxchange.read_aps_32id(fname, sino=sino)
            
            # Set data collection angles as equally spaced between 0-180 degrees.
            theta = tomopy.angles(proj.shape[0])

            # Flat-field correction of raw data.
            proj = tomopy.normalize(proj, flat[15:20], dark[8:10])
            
            center = tomopy.find_center_pc(proj[0], proj[proj.shape[0] - 1])
            print ("Center:", center)

            rec = tomopy.recon(proj, theta, center=center, algorithm='gridrec')
            #rec = tomopy.write_center(proj, theta, dpath=rec_fname, cen_range=[rot_start, rot_end, rot_step], ind=0, mask=True)
    
            # Mask each reconstructed slice with a circle.
            rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

            rec_fname = (folder + 'center_auto' + os.sep + 'data')
            # Write data as stack of TIFs.
            dxchange.write_tiff_stack(rec, fname=rec_fname)    
    except:
        print (folder, 'does not contain the expected file hdf5 file')
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
