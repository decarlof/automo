#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from glob import glob
import os
from os.path import expanduser
home = expanduser("~")

setup(
    name='automo',
    author='Francesco De Carlo, Rafael Vescovi',
    packages=find_packages(),
    version=open('VERSION').read().strip(),
    description = 'Automation for tomography.',
    license='BSD-3',
    platforms='Any',
    scripts=glob('config/automo_*'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: BSD-3',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
)

try:
    print('Do you want to add PATH exporting of automo macro folder to your bashrc? (y/n) ')
    write_bashrc = input()
    if write_bashrc in ['Y', 'y']:
        f = open(os.path.join(home, '.bashrc'), 'a')
        f.write('export PATH=' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'macros') + ':$PATH\n')
        f.close()
except:
    print('I can\'t write into .bashrc when installing with pip. Please append the following line to your .bashrc:')
    print('export PATH=' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'macros') + ':$PATH\n')
    

os.system('export PATH=' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'macros') + ':$PATH')