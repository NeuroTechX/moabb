=====================
API and Main Concepts
=====================

.. figure:: images/architecture.png
   :alt: architecture


There are 4 main concepts in the MOABB: **the datasets**, **the paradigm**, **the
evaluation**, and **the pipelines**. In addition, we offer **statistical**,
**visualization**, **utilities** to simplify the workflow.

And if you want to just run the benchmark, you can use our **benchmark** module that wraps
all the steps in a single function.


Datasets
--------
.. currentmodule:: moabb.datasets

A dataset handles and abstracts low-level access to the data. The
dataset will read data stored locally, in the format in which they have
been downloaded, and will convert them into an MNE raw object. There are
options to pool all the different recording sessions per subject or to
evaluate them separately.

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
    Dreyer2023
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
    Stieger2021
    Liu2024
    Beetl2021_A
    Beetl2021_B

-----------------
ERP/P300 Datasets
-----------------

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
    ErpCore2021_ERN
    ErpCore2021_LRP
    ErpCore2021_MMN
    ErpCore2021_N2pc
    ErpCore2021_N170
    ErpCore2021_N400
    ErpCore2021_P3
    Kojima2024A

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
    Hinss2021
    Rodrigues2017

-----------------
Compound Datasets
-----------------
.. currentmodule:: moabb.datasets.compound_dataset

.. autosummary::
    :toctree: generated/
    :template: class.rst

    BI2014a_Il
    BI2014b_Il
    BI2015a_Il
    BI2015b_Il
    Cattan2019_VR_Il
    BI_Il

------------
Base & Utils
------------
.. currentmodule:: moabb.datasets

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BaseDataset
    base.BaseBIDSDataset
    base.LocalBIDSDataset
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
    utils.plot_datasets_grid
    utils.plot_datasets_cluster

Paradigm
--------
.. currentmodule:: moabb.paradigms

A paradigm defines how the raw data will be converted to trials ready to
be processed by a decoding algorithm. This is a function of the paradigm
used, i.e. in motor imagery one can have two-class, multi-class, or
continuous paradigms; similarly, different preprocessing is necessary
for ERP vs ERD paradigms.

-----------------------
Motor Imagery Paradigms
-----------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MotorImagery
    LeftRightImagery

    FilterBankLeftRightImagery
    FilterBankMotorImagery

--------------
P300 Paradigms
--------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

     SinglePass
     P300

---------------
SSVEP Paradigms
---------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    SSVEP
    FilterBankSSVEP

---------------
c-VEP Paradigms
---------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    CVEP
    FilterBankCVEP

-----------------------
Resting state Paradigms
-----------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    RestingStateToP300Adapter

-----------------------------------
Fixed Interval Windows Processings
-----------------------------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    FixedIntervalWindowsProcessing
    FilterBankFixedIntervalWindowsProcessing

------------
Base & Utils
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    motor_imagery.BaseMotorImagery
    motor_imagery.SinglePass
    motor_imagery.FilterBank
    p300.BaseP300
    ssvep.BaseSSVEP
    BaseFixedIntervalWindowsProcessing
    base.BaseParadigm
    base.BaseProcessing

Evaluations
-----------
.. currentmodule:: moabb.evaluations

An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) – it can be
either within-recording-session accuracy, across-session within-subject
accuracy, across-subject accuracy, or other transfer learning settings.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    WithinSessionEvaluation
    CrossSessionEvaluation
    CrossSubjectEvaluation

.. autosummary::
    :toctree: generated/
    :template: class.rst

    WithinSessionSplitter
    CrossSessionSplitter
    CrossSubjectSplitter

------------
Base & Utils
------------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    base.BaseEvaluation

Pipelines
---------
.. currentmodule:: moabb.pipelines

Pipeline defines all steps required by an algorithm to obtain
predictions. Pipelines are typically a chain of sklearn compatible
transformers and end with a sklearn compatible estimator. See
`Pipelines <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`__
for more info.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    features.LogVariance
    features.FM
    features.ExtendedSSVEPSignal
    features.AugmentedDataset
    features.StandardScaler_Epoch
    csp.TRCSP
    classification.SSVEP_CCA
    classification.SSVEP_TRCA
    classification.SSVEP_MsetCCA

Statistics, visualization and utilities
---------------------------------------
.. currentmodule:: moabb.analysis

Once an evaluation has been run, the raw results are returned as a
DataFrame. This can be further processed via the following commands to
generate some basic visualization and statistical comparisons:

--------
Plotting
--------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    plotting.score_plot
    plotting.paired_plot
    plotting.summary_plot
    plotting.meta_analysis_plot
    plotting.dataset_bubble_plot

----------
Statistics
----------

.. autosummary::
    :toctree: generated/
    :template: function.rst

    meta_analysis.find_significant_differences
    meta_analysis.compute_dataset_statistics
    meta_analysis.combine_effects
    meta_analysis.combine_pvalues
    meta_analysis.collapse_session_scores

-----
Utils
-----
.. currentmodule:: moabb

.. autosummary::
    :toctree: generated/
    :template: function.rst

    set_log_level
    setup_seed
    set_download_dir
    make_process_pipelines

Benchmark
---------
.. currentmodule:: moabb

The benchmark module wraps all the steps in a single function. It
downloads the data, runs the benchmark, and returns the results. It is
the easiest way to run a benchmark.

.. code:: python

    from moabb import benchmark

    results = benchmark(
        pipelines="./pipelines",
        evaluations=["WithinSession"],
        paradigms=["LeftRightImagery"],
        include_datasets=[BNCI2014_001(), PhysionetMI()],
        exclude_datasets=None,
        results="./results/",
        overwrite=True,
        plot=True,
        output="./benchmark/",
        n_jobs=-1,
    )

.. autosummary::
    :toctree: generated/
    :template: function.rst

    benchmark
