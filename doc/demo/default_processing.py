#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function

import sys
import automo


if __name__ == "__main__":

    
    automo.move(sys.argv[1:])
    automo.run_test(sys.argv[1:])
