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
    Dreyer2023A
    Dreyer2023B
    Dreyer2023C
    Lee2019_MI
    GrosseWentrup2009
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

    BI2012
    BI2013a
    BI2014a
    BI2014b
    BI2015a
    BI2015b
    Cattan2019_VR
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

    Kalunga2016
    Nakanishi2015
    Wang2016
    MAMEM1
    MAMEM2
    MAMEM3
    Lee2019_SSVEP



--------------
c-VEP Datasets
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Thielen2015
    Thielen2021
    CastillosBurstVEP40
    CastillosBurstVEP100
    CastillosCVEP40
    CastillosCVEP100

----------------------
Resting State Datasets
----------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Cattan2019_PHMD


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

    BI2014a_Il
    BI2014b_Il
    BI2015a_Il
    BI2015b_Il
    Cattan2019_VR_Il
    BI_Il
