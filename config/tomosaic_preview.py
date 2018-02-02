#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo with Tomosaic.
"""

import tomosaic
from tomosaic import *
from tomosaic.misc.misc import read_data_adaptive
import glob, os
import numpy as np
from mosaic_meta import *
import dxchange
import tomopy
import sys
import argparse

# ==========================================
frame = 0
method = 'pyramid'
margin = 50
src_folder = 'data_raw_1x'
blend_options = {'depth': 7,
                 'blur': 0.4}
# ==========================================

def preprocess(dat, blur=None):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat

def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", help="folder where the H5 files are located",default='data_raw_1x')
    parser.add_argument("--frame", help="frame to preview",default=0)
    args = parser.parse_args()
    src_folder = args.src_folder
    frame = int(args.frame)

    root = os.getcwd()
    os.chdir(src_folder)
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
    buff = np.zeros([1, 1])
    for (y, x), value in np.ndenumerate(file_grid):
        if value != None:
            prj, flt, drk, _ = read_data_adaptive(value, proj=(frame, frame + 1))
            prj = tomopy.normalize(prj, flt, drk)
            prj = preprocess(np.copy(prj))
            buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :], method=method)
            print(y, x)

    os.chdir(root)
    dxchange.write_tiff(buff, 'panos/{}_norm'.format(frame), dtype='float32', overwrite=True)


if __name__ == "__main__":
    main(sys.argv[1:])