Default Processing 
==================

Automo directory generation and default processing 
(:download:`default_processing.py<../../../doc/demo/default_processing.py>`).


Pre-requisites
++++++++++++++

Default Processing relies on the following configuration:

Before using :download:`default_processing.py<../../../doc/demo/default_processing.py>` you need to 
create `~/.tomo/automo.ini <https://github.com/decarlof/automo/blob/master/config/automo.ini>`__
configuration file and default auto test python scripts to run on the data matching the names
defined in the `python_proc` label. The names and number of `python_proc` function is arbitrary.

For :download:`automo.ini<../../../config/automo.ini>` you can use 
:download:`center.py<../../../config/center.py>`, :download:`preview.py<../../../config/preview.py>`,
and :download:`recon.py<../../../config/recon.py>`.



.. literalinclude:: ../../../doc/demo/default_processing.py    :tab-width: 4    :linenos:    :language: guess

