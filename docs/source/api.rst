=====================
API and Main Concepts
=====================

.. raw:: html

   <div class="api-hero">

.. figure:: images/architecture.svg
   :alt: Concept flow in MOABB
   :class: api-architecture-diagram

   Datasets and Paradigms define the problem; Evaluations and Pipelines
   define the measurement.

.. raw:: html

   <p class="api-intro">
   There are 4 main concepts in the MOABB:
   <strong class="concept-dataset">the datasets</strong>,
   <strong class="concept-paradigm">the paradigms</strong>,
   <strong class="concept-evaluation">the evaluations</strong>, and
   <strong class="concept-pipeline">the pipelines</strong>.
   In addition, we offer <strong>statistical</strong>,
   <strong>visualization</strong>, <strong>utilities</strong> to simplify the workflow.
   </p>
   <p class="api-intro">
   And if you want to just run the benchmark, you can use our
   <strong>benchmark</strong> module that wraps all the steps in a single function.
   </p>
   </div>


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
    BNCI2003_004
    BNCI2014_001
    BNCI2014_002
    BNCI2014_004
    BNCI2015_001
    BNCI2015_004
    BNCI2019_001
    BNCI2020_001
    BNCI2022_001
    BNCI2024_001
    BNCI2025_001
    BNCI2025_002
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
    Brandl2020
    Chang2025
    Forenzo2023
    Gao2026
    GuttmannFlury2025_MI
    HefmiIch2025
    Jeong2020
    Kaya2018
    Kumar2024
    Liu2025
    Ma2020
    Rozado2015
    Tavakolan2017
    TrianaGuzman2024
    Wairagkar2018
    Wu2020
    Yang2025
    Yi2025
    Zhang2017
    Zhou2020
    Zuo2025

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
    BNCI2015_006
    BNCI2015_007
    BNCI2015_008
    BNCI2015_009
    BNCI2015_010
    BNCI2015_012
    BNCI2015_013
    BNCI2016_002
    BNCI2020_002
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
    RomaniBF2025ERP
    Kojima2024A
    Kojima2024B
    Lee2021Mobile_ERP
    Chailloux2020
    GuttmannFlury2025_P300
    Kaneshiro2015
    Lee2024_AC
    Lee2024_BS
    Lee2024_DL
    Lee2024_EL
    Lee2024_TV
    Mainsah2025_A
    Mainsah2025_B
    Mainsah2025_C
    Mainsah2025_D
    Mainsah2025_E
    Mainsah2025_F
    Mainsah2025_G
    Mainsah2025_H
    Mainsah2025_I
    Mainsah2025_J
    Mainsah2025_K
    Mainsah2025_L
    Mainsah2025_M
    Mainsah2025_N
    Mainsah2025_O
    Mainsah2025_P
    Mainsah2025_Q
    Mainsah2025_R
    Mainsah2025_S1
    Mainsah2025_S2
    Simoes2020
    Speier2017
    Zhang2025
    Zheng2020

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
    Chen2017SingleFlicker
    Dong2023
    Han2024Fatigue
    Kim2025BetaRange
    Lee2021Mobile_SSVEP
    Liu2020BETA
    Liu2022EldBETA
    Wang2021Combined
    GuttmannFlury2025_SSVEP

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
    MartinezCagigal2023Checker
    MartinezCagigal2023Pary

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

---------
Utilities
---------
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

Paradigms
---------
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
Resting State Paradigms
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

---------
Utilities
---------

.. autosummary::
    :toctree: generated/
    :template: class.rst

    motor_imagery.BaseMotorImagery
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
    WithinSubjectSplitter
    CrossSessionSplitter
    CrossSubjectSplitter

---------
Utilities
---------

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
    classification.SSVEP_itCCA
    classification.SSVEP_eCCA
    classification.SSVEP_TRCA_R
    classification.SSVEP_SSCOR
    classification.SSVEP_TDCA

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

.. admonition:: Minimal benchmark example

   .. code-block:: python

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
