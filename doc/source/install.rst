==================
Install directions
==================

This section covers the basics of how to download and install `auTomo <https://github.com/decarlof/automo>`_.

.. contents:: Contents:
   :local:


Installing from conda
=====================

auTomo has an conda install script that does all the job. Please follow the script:

    git clone https://github.com/decarlof/automo.git automo
    cd automo
    conda build .
    conda install --use-local automo

Installing from source
======================

To install from source some other steps are needed. Please follow the script:

    git clone https://github.com/decarlof/automo.git automo
    cd automo
    python setup.py install
    mkdir ~/.automo
    cp config/* ~/.automo

Creating new auTomo robos
=========================

After installing (either via conda or source) you will find the automo robos at `~/.automo/automo.ini`

The initial configuration will be exactly as `automo.ini<../../config/automo.ini>`

You can create new robo process by adding at the `~/.automo/` folder the corresponded script. This script needs
to be self-contained and run into a single file (this is the information auTomo will provide to the robo).
This process also needs to be added to the configuration file and default auto test python scripts
to run on the data matching the names defined in the `python_proc` label.
Some examples are: :download:`center.py<../../config/center.py>`, :download:`preview.py<../../config/preview.py>`,
and :download:`recon.py<../../config/recon.py>`.
