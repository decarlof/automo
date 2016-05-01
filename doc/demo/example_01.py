#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function

import sys
import automo


if __name__ == "__main__":

    
    automo.mv_tomo(sys.argv[1:])
    automo.run_tomo(sys.argv[1:])
