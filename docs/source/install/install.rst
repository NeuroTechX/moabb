.. _installation:

================
Installation
================

MOABB is written in Python 3, specifically for version 3.8 or above.

The package is distributed via Python package index (`PyPI <https://pypi.org/project/moabb>`__), and you can access the
source code via `Github <https://github.com/NeuroTechX/moabb>`__ repository. The pre-built Docker images using the core
library and all optional dependencies are available on `DockerHub <https://hub.docker.com/r/baristimunha/moabb>`__.

There are different ways to install MOABB:


.. grid:: 3

    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

            Install via ``pip``

        .. rst-class:: card-subtitle text-muted mt-0

            For Beginners

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: ../../source/_static/moabb_install.png
           :alt: MOABB Installer with pip

        **New to Python?** Use our standalone installers that include
        everything to get you started!
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        .. button-ref:: install_pip
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            Installing from PyPI


    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

           Building from src and the dev env

        .. rst-class:: card-subtitle text-muted mt-0

            For Advanced Users

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: https://mne.tools/stable/_images/mne_installer_console.png
           :alt: Terminal Window

        **Already familiar with Python?**
        Follow our setup instructions for building from Github and start to contribute!
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        .. button-ref:: install_source
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            From src and dev env

    .. grid-item-card::
        :text-align: center

        .. rst-class:: font-weight-bold mb-0

           Using our docker environment

        .. rst-class:: card-subtitle text-muted mt-0

            For Advanced Users

        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        .. image:: https://www.docker.com/wp-content/uploads/2022/03/Moby-logo.png
           :alt: Terminal Docker

        **Already familiar with Docker?**
        Follow our setup instructions for using your docker image!
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        .. button-ref:: using_docker
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold

            Using the docker image


.. toctree::
    :hidden:

    install_pip
    install_source
    using_docker
