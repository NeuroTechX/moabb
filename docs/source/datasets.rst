========
Datasets
========

.. automodule:: moabb.datasets

.. currentmodule:: moabb.datasets

----------------------
Motor Imagery Datasets
----------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    AlexMI
    BNCI2014001
    BNCI2014002
    BNCI2014004
    BNCI2015001
    BNCI2015004
    Cho2017
    Lee2019_MI
    MunichMI
    Ofner2017
    PhysionetMI
    Schirrmeister2017
    Shin2017A
    Shin2017B
    Weibo2014
    Zhou2016


------------
ERP Datasets
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bi2012a
    bi2013a
    bi2014a
    bi2014b
    bi2015a
    bi2015b
    BNCI2014008
    BNCI2014009
    BNCI2015003
    DemonsP300
    EPFLP300
    Huebner2017
    Huebner2018
    Lee2019_ERP
    Sosulski2019


--------------
SSVEP Datasets
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SSVEPExo
    Nakanishi2015
    Wang2016
    MAMEM1
    MAMEM2
    MAMEM3
    Lee2019_SSVEP


------------
Base & Utils
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BaseDataset


.. autosummary::
    :toctree: generated/
    :template: function.rst

    download.data_path
    download.data_dl
    download.fs_issue_request
    download.fs_get_file_list
    download.fs_get_file_hash
    download.fs_get_file_id
    download.fs_get_file_name
    utils.dataset_search
    utils.find_intersecting_channels
