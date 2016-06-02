#!/bin/bash
$PYTHON setup.py install

# Add more build steps here, if they are necessary.

mkdir -p ~/.automo
cp config/* ~/.automo

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
