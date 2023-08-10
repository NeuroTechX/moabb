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
    BNCI2014_001
    BNCI2014_002
    BNCI2014_004
    BNCI2015_001
    BNCI2015_004
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

    bi2012
    bi2013a
    bi2014a
    bi2014b
    bi2015a
    bi2015b
    VirtualReality
    BNCI2014_008
    BNCI2014_009
    BNCI2015_003
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


----------------------
Resting State Datasets
----------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Cattan2019


------------
Base & Utils
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BaseDataset
    base.CacheConfig
    fake.FakeDataset
    fake.FakeVirtualRealityDataset


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


====================
Compound Datasets
====================

.. automodule:: moabb.datasets.compound_dataset

.. currentmodule:: moabb.datasets.compound_dataset

------------
ERP Datasets
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    bi2014a_il
    bi2014b_il
    bi2015a_il
    bi2015b_il
    VirtualReality_il
    biIlliteracy
