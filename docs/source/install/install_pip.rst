.. _install_pip:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

MOABB can be installed via pip from `PyPI <https://pypi.org/project/moabb>`__.

.. warning::
    MOABB requires **Python 3.11 or newer**.

.. note::
    We recommend the most updated version of pip to install from PyPI.

Below are the installation commands for the most common use cases.

.. code-block:: bash

   pip install moabb

MOABB can also be installed with sets of optional dependencies to enable the deep learning modules, the tracker of the carbon emission module or the documentation environment.

For example, to install all the optional dependencies.

.. code-block:: bash

   pip install moabb[deepleaning,carbonemission,docs]

See the issue at Github if you have a problem.
