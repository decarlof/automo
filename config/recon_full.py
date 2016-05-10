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
import automo.util as util
import argparse
import ConfigParser
from os.path import expanduser


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="existing hdf5 file name")
    parser.add_argument("slice_start", help="recon slice start; for full rec enter -1")
    parser.add_argument("slice_end", help="recon slice end; for full rec enter -1")
    parser.add_argument("slice_step", help="recon slice step; for full rec enter -1")
    parser.add_argument("rot_center", help="rotation center; for auto center enter -1")
#    parser.add_argument("save_dir", help="relative save directory")
    args = parser.parse_args()

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    fname = args.file_name

    array_dims = util.h5group_dims(fname)

    print("Data: ", array_dims)
    # Select the sinogram range to reconstruct.
    sino_start = int(args.slice_start)
    sino_end = int(args.slice_end)
    sino_step = int(args.slice_step)
    
    sino_range = range(1, array_dims[1])
    if not ((sino_start < sino_end) and sino_start in sino_range and sino_end in sino_range):
        sino_start = 1
        sino_end = array_dims[1]

    sino_step_range = range(0, array_dims[1]-1)
    if not sino_step in sino_step_range:
        sino_step =1
    
    sino = [sino_start, sino_end, sino_step]
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

            # Select the rotation center range.
            rot_center = int(args.rot_center)
            rot_center_range = range(0, array_dims[2])
            if not (rot_center in rot_center_range):
                rot_center = tomopy.find_center_pc(proj[0], proj[proj.shape[0] - 1])
                print("Auto Center:", rot_center)
            else:
                print("Manual Center:", rot_center)

            print("Recon:", sino)
            rec = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')
    
            # Mask each reconstructed slice with a circle.
            rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

            # Write data as stack of TIFs.
            rec_fname = (folder + 'recon_full' + os.sep + 'data')
            print("Rec folder: ", rec_fname)
            dxchange.write_tiff_stack(rec, fname=rec_fname, overwrite=True)    
            print("#################################")
    except:
        print(folder, 'does not contain the expected file hdf5 file')
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
