#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo example to generage sample previews.
"""

from __future__ import print_function

import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob

import automo.util as util

##PROBABLY SHOULD GO TO A STANDARD OPERATIONS FILE.
def get_yz_slice(recon_folder, chunk_size=50, slice_y=1000):

    filelist = glob.glob(os.path.join(recon_folder, 'recon*.tiff'))
    inds = []
    digit = None
    for i in filelist:
        i = os.path.split(i)[-1]
        regex = re.compile(r'\d+')
        a = regex.findall(i)[0]
        if digit is None:
            digit = len(a)
        inds.append(int(a))
    chunks = []
    chunk_st = np.min(inds)
    chunk_end = chunk_st + chunk_size
    sino_end = np.max(inds)

    while chunk_end < sino_end:
        chunks.append((chunk_st, chunk_end))
        chunk_st = chunk_end
        chunk_end += chunk_size
    chunks.append((chunk_st, sino_end))

    recon_stack = dxchange.read_tiff_stack(filelist[0], range(chunks[0][0], chunks[0][1]), digit)
    slice = recon_stack[:, slice_y, :]
    for (chunk_st, chunk_end) in chunks[1:]:
        a = dxchange.read_tiff_stack(filelist[0], range(chunk_st, chunk_end), digit)
        slice = np.append(slice, a[:, slice_y, :], axis=0)

    return slice

def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("rec_folder", help="existing recon foldername",default='auto')
    args = parser.parse_args()
    
    fname = args.file_name

    if fname == 'auto':
        rec_folders = sorted(glob('/recon*'))
        fname = rec_folders[-1] 
        print ('Autofilename =' + fname)

    if os.path.isdir(fname):

        slice1 = get_yz_slice(fname, chunk_size=50, slice_y=1000)
        dxchange.write_tiff(slice1, os.path.join(fname+'_preview', 'yz_cs.tiff'), dtype='float32')

if __name__ == "__main__":
    main(sys.argv[1:])