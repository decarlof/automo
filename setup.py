#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from glob import glob
setup(
    name='automo',
    author='Francesco De Carlo, Rafael Vescovi',
    packages=find_packages(),
    version=open('VERSION').read().strip(),
    description = 'Automation for tomography.',
    license='BSD-3',
    platforms='Any',
    scripts=glob('macros/automo_*'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: BSD-3',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
)
