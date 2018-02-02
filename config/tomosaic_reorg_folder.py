#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AuTomo with Tomosaic.
"""

import tomosaic
import time
from mpi4py import MPI
# from mosaic_meta import *
import os
import sys
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", help="downsample levels, separated by comma",default='1,2,4')
    args = parser.parse_args()
    ds = args.ds
    ds = [int(i) for i in ds.split(',')]

    t0 = time.time()
    tomosaic.reorganize_dir(file_list, raw_ds=ds)

    print('Total time: {}s'.format(time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1:])