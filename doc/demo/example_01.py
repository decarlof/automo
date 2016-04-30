#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function
import automo
from os.path import expanduser
import ConfigParser
import os


if __name__ == "__main__":

    home = expanduser("~")
    tomo = os.path.join(home, '.tomo/automo.ini')
    cf = ConfigParser.ConfigParser()
    cf.read(tomo)

    pdir = cf.get('settings', 'python_proc_dir')
    processes = cf.get('settings', 'python_proc')
    processes = processes.split(', ')
    h5_fname = cf.get('settings', 'h5_fname')
       
    automo.create_tomo(sys.argv[1:])
