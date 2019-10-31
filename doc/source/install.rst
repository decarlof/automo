==================
Install directions
==================

This section covers the basics of how to download and install `auTomo <https://github.com/decarlof/automo>`_.

.. contents:: Contents:
   :local:

   
Installing from source (recommended)
====================================

Installing from source can be done easily by running the setup script:
::

  git clone https://github.com/decarlof/automo.git automo
  cd automo
  python setup.py install

The script will ask whether you want to add a line in your ``.bashrc`` file so that bash will automatically 
add the ``macros`` folder in the Automo source directory to your ``$PATH`` variable. This is necessary if you
would like to use the feature of calling Automo script directly from bash command line. If you prefer to
have the scripts, the configuration file (``automo.ini``), and the parameter setting file (``automo_params.csv``)
somewhere else, you need to manually copy them
there. For example, to move the files to ``~/.automo``, follow the above commands by
::

    mkdir ~/.automo
    cp macros ~/.automo
    export PATH=~/.automo:$PATH

Subsequently, add the last line ``export PATH=~/.automo:$PATH`` to your ``~/.bashrc``.
    
    
Installing from conda
=====================

auTomo has an conda install script that does all the job. Please follow the script:
::

    git clone https://github.com/decarlof/automo.git automo
    cd automo
    conda build .
    conda install --use-local automo

