#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function

import sys
import automo.robo as robo


if __name__ == "__main__":

    
    robo.move(sys.argv[1:])
    robo.test(sys.argv[1:])
