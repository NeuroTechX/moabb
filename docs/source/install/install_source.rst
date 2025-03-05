.. _install_source:

Installing from sources
~~~~~~~~~~~~~~~~~~~~

If you want to test features under development or contribute to the library, or if you want to test the new tools that have been tested in moabb and not released yet, this is the right tutorial for you!

.. note::

   If you are only trying to install MOABB, we recommend using the pip installation `Installation <https://neurotechx.github.io/moabb/install/install_pip.html#install-pip>`__ for details on that.

.. _system-level:

Clone the repository from GitHub
--------------------------------------------------

The first thing you should do is clone the MOABB repository to your computer and enter inside the repository.

.. code-block:: bash

   git clone https://github.com/neurotechx/moabb && cd moabb

You should now be in the root directory of the MOABB repository.

Installing Moabb from the source
--------------------------------------------------------------------------------------------------------------------------------

If you want to only install Moabb from source once and not do any development
work, then the recommended way to build and install is to use ``pip``::

For the latest development version, directly from GitHub:

.. code-block:: bash

   pip install https://github.com/NeuroTechX/moabb/archive/refs/heads/develop.zip

If you have a local clone of the MOABB git repository:

.. code-block:: bash

   pip install .

You can also install MOABB in editable mode (i.e. changes to the source code).

Building MOABB from source with the development environment
----------------------------------------------------------------------------------------

If you want to build from source to work on MOABB itself, then follow these steps:

1. Install pip on your system follow the tutorial: https://pip.pypa.io/en/stable/installation/.

2. You will need to run this command in the project directory:

.. code-block:: console

   pip install -e .

3. If you want to install with an optional dependency

.. code-block:: console

   pip install -e .[deeplearning,carbonemission,docs,optuna,tests]

For a full list of dependencies, see the pyproject.toml file.

To contribute with a library you must install ``pre-commit``, follow this tutorial   `Installation Pre-Commit <https://pre-commit.com/#install>`__. To more details to become a contributors, see
`contributors' guidelines <https://github.com/NeuroTechX/moabb/blob/master/CONTRIBUTING.md>`__.
for a detailed explanation.


Testing if your installation is working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that MOABB is installed and running correctly, run the following command:

.. code-block:: console

   pytest moabb/tests --verbose

For more information, please see the contributors' guidelines.
