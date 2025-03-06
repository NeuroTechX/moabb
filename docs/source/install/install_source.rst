.. _install_source:

Installation from Sources
===========================

This guide is intended for users who want to test experimental features or contribute to MOABB’s development. If you only need the stable release, please refer to our
`pip installation guide <https://neurotechx.github.io/moabb/install/install_pip.html#install-pip>`__.

.. note::
   For a straightforward MOABB installation (without development), please see the pip installation guide.

Prerequisites
-------------
Before proceeding, ensure that you have the following:

- A working installation of Python.
- Pip installed on your system (see the `pip installation guide <https://pip.pypa.io/en/stable/installation/>`__).

Cloning the Repository
----------------------
First, clone the MOABB repository from GitHub and navigate into the project directory:

.. code-block:: bash

   git clone https://github.com/neurotechx/moabb.git
   cd moabb

Installing MOABB from the Source
--------------------------------
If you wish to install MOABB for usage without modifying the code, use one of the following methods:

- **Install the Latest Development Version (from GitHub):**

  .. code-block:: bash

     pip install https://github.com/NeuroTechX/moabb/archive/refs/heads/develop.zip

- **Install from a Local Clone:**

  .. code-block:: bash

     pip install .

- **Editable Installation:**

  This mode installs MOABB so that any local changes are immediately available:

  .. code-block:: bash

     pip install -e .

Setting Up a Development Environment
--------------------------------------
For contributors or those who want to work on MOABB’s codebase, follow these steps:

1. **Ensure pip is installed.**
   (Refer to the `pip installation guide <https://pip.pypa.io/en/stable/installation/>`__.)

2. **Basic Editable Installation (without optional dependencies):**
   In the project directory, run:

   .. code-block:: bash

      pip install -e .

3. **Editable Installation with Optional Dependencies:**
   If you require additional features (e.g., deep learning, testing), install with:

   .. code-block:: bash

      pip install -e .[deeplearning,carbonemission,docs,optuna,tests,external]

   For a complete list of optional dependencies, consult the `pyproject.toml` file.

4. **Setup Pre-Commit Hooks:**
   To help maintain code quality, install the ``pre-commit`` tool by following the
   `Pre-Commit Installation guide <https://pre-commit.com/#install>`__. For further instructions on contributing, see the
   `Contributors Guidelines <https://github.com/NeuroTechX/moabb/blob/master/CONTRIBUTING.md>`__.

Verifying the Installation
--------------------------
To ensure that MOABB is installed and functioning correctly, run:

.. code-block:: console

   pytest moabb/tests --verbose

For more details or troubleshooting, please refer to the
`Contributors Guidelines <https://github.com/NeuroTechX/moabb/blob/master/CONTRIBUTING.md>`__.
