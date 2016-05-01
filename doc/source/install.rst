==================
Install directions
==================

This section covers the basics of how to download and install `Automo <https://github.com/decarlof/automo>`_.

.. contents:: Contents:
   :local:


Pre-requisites
==============

Before using `Automo <https://github.com/decarlof/automo>`_  you need to 
create `~/.tomo/automo.ini <https://github.com/decarlof/automo/blob/master/config/automo.ini>`__
configuration file and default auto test python scripts to run on the data matching the names
defined in the `python_proc` label. The names and number of `python_proc` function is arbitrary.

For :download:`automo.ini<../../config/automo.ini>` you can use 
:download:`center.py<../../config/center.py>`, :download:`preview.py<../../config/preview.py>`,
and :download:`recon.py<../../config/recon.py>`.


Installing from source
======================

Clone the `Automo <https://github.com/decarlof/automo>`_  
from `GitHub <https://github.com>`_ repository::

    git clone https://github.com/decarlof/automo.git automo

then::

    cd automo
    python setup.py install

