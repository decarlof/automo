#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function

import os
import automo
import ConfigParser
from os.path import expanduser


if __name__ == "__main__":

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    pdir = cf.get('settings', 'python_proc_dir')
    processes = cf.get('settings', 'python_proc')
    processes = processes.split(', ')
    h5_fname = cf.get('settings', 'h5_fname')
       
    automo.run_tomo(sys.argv[1:])
