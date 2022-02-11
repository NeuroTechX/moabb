.. _whats_new:

.. currentmodule:: moabb

What's new
==========

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes
.. _current:


Develop branch
----------------

Enhancements
~~~~~~~~~~~~

- Progress bars, pooch, tqdm (:gh:`258` by `Divyesh Narayanan`_ and `Sylvain Chevallier`_)
- Adding Test and Example for set_download_dir (:gh:`249` by `Divyesh Narayanan`_)

Bugs
~~~~

- Removing dependency on mne method for PhysionetMI data downloading, renaming runs (:gh:`257` by `Divyesh Narayanan`_)
- Correcting events management in Schirrmeister2017, renaming session and run (:gh:`255` by `Pierre Guetschel`_ and `Sylvain Chevallier`_)
- Switch session and runs in MAMEM1, 2 and 3 to avoid error in WithinSessionEvaluation (:gh:`256` by `Sylvain Chevallier`_)
- Correct doctstrings for the documentation, incuding Lee2017 (:gh:`256` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~

- None


Version - 0.4.4  (Stable - PyPi)
---------------

Enhancements
~~~~~~~~~~~~

- Add TRCA algorithm for SSVEP (:gh:`238` by `Ludovic Darmet`_)

Bugs
~~~~

- Remove unused argument from dataset_search (:gh:`243` by `Divyesh Narayanan`_)
- Remove MNE call to `_fetch_dataset` and use MOABB `_fetch_file` (:gh:`235` by `Jan Sosulski`_)
- Correct doc formatting (:gh:`232` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~

- Minimum supported Python version is now 3.7
- MOABB now depends on scikit-learn >= 1.0


Version - 0.4.3
----------------

Enhancements
~~~~~~~~~~~~

- Rewrite Lee2019 to add P300 and SSVEP datasets (:gh:`217` by `Pierre Guetschel`_)

Bugs
~~~~

- Avoid information leakage for MNE Epochs pipelines in evaluation (:gh:`222` by `Sylvain Chevallier`_)
- Correct error in set_download_dir (:gh:`225` by `Sylvain Chevallier`_)
- Ensure that channel order is consistent across dataset when channel argument is specified in paradigm (:gh:`229` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~

- ch_names argument added to init of moabb.datasets.fake.FakeDataset (:gh:`229` by `Sylvain Chevallier`_)

Version - 0.4.2
---------------

Enhancements
~~~~~~~~~~~~
- None

Bugs
~~~~
- Correct error when downloading Weibo dataset  (:gh:`212` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~
- None

Version - 0.4.1
---------------

Enhancements
~~~~~~~~~~~~
- None

Bugs
~~~~
- Correct path error for first time launch (:gh:`204` by `Sylvain Chevallier`_)
- Fix optional dependencies issues for PyPi (:gh:`205` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~
- Remove update_path on all datasets, `update_path` parameter in `dataset.data_path()` is deprecated (:gh:`207` by `Sylvain Chevallier`_)

Version - 0.4.0
---------------

Enhancements
~~~~~~~~~~~~
- Implementation for learning curves (:gh:`155` by `Jan Sosulski`_)
- Adding Neiry Demons P300 dataset (:gh:`156` by `Vladislav Goncharenko`_)
- Coloredlogs removal (:gh:`163` by `Vladislav Goncharenko`_)
- Update for README (:gh:`164` by `Vladislav Goncharenko`_ and `Sylvain Chevallier`_)
- Test all relevant python versions in Github Actions CI (:gh:`167` by `Vladislav Goncharenko`_)
- Adding motor imagery part of the Lee2019 dataset (:gh:`170` by `Ali Abdul Hussain`_)
- CI: deploy docs from CI pipeline  (:gh:`124` by `Erik Bjäreholt`_, `Divyesh Narayanan`_ and `Sylvain Chevallier`_)
- Remove dependencies: WFDB and pyunpack (:gh:`180` and :gh:`188` by `Sylvain Chevallier`_)
- Add support for FigShare API (:gh:`188` by `Sylvain Chevallier`_)
- New download backend function relying on Pooch, handling FTP, HTTP and HTTPS (:gh:`188` by `Sylvain Chevallier`_)
- Complete rework of examples and tutorial (:gh:`188` by `Sylvain Chevallier`_)
- Change default storage location for results: instead of moabb source code directory it is now stored in mne_data (:gh:`188` by `Sylvain Chevallier`_)
- Major update of test (:gh:`188` by `Sylvain Chevallier`_)
- Adding troubleshooting and badges in README (:gh:`189` by `Jan Sosulski`_ and `Sylvain Chevallier`_)
- Use MNE epoch in evaluation (:gh:`192` by `Sylvain Chevallier`_)
- Allow changing of storage location (:gh:`192` by `Divyesh Narayanan`_ and `Sylvain Chevallier`_)
- Deploy docs on moabb.github.io (:gh:`196` by `Sylvain Chevallier`_)
- Broadening subject_list type for :func:`moabb.datasets.BaseDataset` (:gh:`198` by `Sylvain Chevallier`_)
- Adding this what's new (:gh:`200` by `Sylvain Chevallier`_)
- Improving cache usage and save computation time in CI (:gh:`200` by `Sylvain Chevallier`_)
- Rewrite Lee2019 to add P300 and SSVEP datasets (:gh:`217` by `Pierre Guetschel`_)


Bugs
~~~~
- Restore basic logging (:gh:`177` by `Jan Sosulski`_)
- Correct wrong type of results dataframe columns (:gh:`188` by `Sylvain Chevallier`_)
- Add ``accept`` arg to acknowledge licence for :func:`moabb.datasets.Shin2017A` and :func:`moabb.datasets.Shin2017B` (:gh:`201` by `Sylvain Chevallier`_)

API changes
~~~~~~~~~~~
- Drop `update_path` from moabb.download.data_path and moabb.download.data_dl


Version 0.3.0
----------------

Enhancements
~~~~~~~~~~~~
- Expose sklearn error_score parameter (:gh:`70` by `Jan Sosulski`_)
- Adds a ``unit_factor`` attribute to base_paradigms (:gh:`72` by `Jan Sosulski`_)
- Allow event lists in P300 paradigm (:gh:`83` by `Jan Sosulski`_)
- Return epochs instead of np.ndarray in process_raw (:gh:`86` by `Jan Sosulski`_)
- Set path for hdf5 files (:gh:`92` by `Jan Sosulski`_)
- Update to MNE 0.21  (:gh:`101` by `Ramiro Gatti`_ and `Sylvain Chevallier`_)
- Adding a baseline correction (:gh:`115` by `Ramiro Gatti`_)
- Adding SSVEP datasets: MAMEM1, MAMEM2, MAMEM3, Nakanishi2015, Wang2016, (:gh:`118` by `Sylvain Chevallier`_, `Quentin Barthelemy`_, and `Divyesh Narayanan`_)
- Switch to GitHub Actions (:gh:`124` by `Erik Bjäreholt`_)
- Allow recording of additional scores/parameters/metrics in evaluation (:gh:`127` and :gh:`128` by `Jan Sosulski`_)
- Fix Ofner2017 and PhysionetMI annotations (:gh:`135` by `Ali Abdul Hussain`_)
- Adding Graz workshop tutorials (:gh:`130` and :gh:`137` by `Sylvain Chevallier`_ and `Lucas Custódio`_)
- Adding pre-commit configuration using isort, black and flake8 (:gh:`140` by `Vladislav Goncharenko`_)
- style: format Python code with black  (:gh:`147` by `Erik Bjäreholt`_)
- Switching to Poetry dependency management (:gh:`150` by `Vladislav Goncharenko`_)
- Using Prettier to format md and yml files (:gh:`151` by `Vladislav Goncharenko`_)


Bugs
~~~~
- Use stim_channels or check annotation when loading files in Paradigm  (:gh:`72` by `Jan Sosulski`_)
- Correct MNE issues (:gh:`76` by `Sylvain Chevallier`_)
- Fix capitalization in channel names of cho dataset  (:gh:`90` by `Jan Sosulski`_)
- Correct failling CI tests (:gh:`100` by `Sylvain Chevallier`_)
- Fix EPFL dataset flat signal sections and wrong scaling (:gh:`104` and :gh:`96` by  `Jan Sosulski`_)
- Fix schirrmeister dataset for Python3.8 (:gh:`105` by `Robin Schirrmeister`_)
- Correct event detection problem and duplicate event error (:gh:`106` by `Sylvain Chevallier`_)
- Fix channel selection in paradigm (:gh:`108` by `Sylvain Chevallier`_)
- Fix upperlimb Ofner2017 error and gdf import problem (:gh:`111` and :gh:`110` by `Sylvain Chevallier`_)
- Fix event_id in events_from_annotations missed (:gh:`112` by `Ramiro Gatti`_)
- Fix h5py>=3.0 compatibility issue (:gh:`138` by `Mohammad Mostafa Farzan`_)
- Python 2 support removal (:gh:`148` by `Vladislav Goncharenko`_)
- Travis-ci config removal (:gh:`149` by `Vladislav Goncharenko`_)

API changes
~~~~~~~~~~~
- None

Version 0.2.1
-------------

Enhancements
~~~~~~~~~~~~
- Add Tikhonov regularized CSP in ``moabb.pipelines.csp`` from the paper (:gh:`60` by `Vinay Jayaram`_)
- update to MNE version 0.19 (:gh:`73` by `Jan Sosulski`_)
- Improve doc building in CI (:gh:`60` by `Sylvain Chevallier`_)

Bugs
~~~~
- Update GigaDB Cho2017 URL (`Pedro L. C. Rodrigues`_ and `Vinay Jayaram`_)
- Fix braininvaders ERP data (`Pedro L. C. Rodrigues`_)
- Replace MNE ``read_montage`` with ``make_standard_montage`` (`Jan Sosulski`_)
- Correct Flake and PEP8 error (`Sylvain Chevallier`_)

API changes
~~~~~~~~~~~
- None


Version 0.2.0
-------------

Enhancements
~~~~~~~~~~~~
- MOABB corresponding to the paper version by `Vinay Jayaram`_ and `Alexandre Barachant`_
- Creating P300 paradigm and BNCI datasets (:gh:`53` by `Pedro L. C. Rodrigues`_)
- Adding EPFL P300 dataset (:gh:`56` by `Pedro L. C. Rodrigues`_)
- Adding BrainInvaders P300 dataset (:gh:`57` by `Pedro L. C. Rodrigues`_)
- Creating SSVEP paradigm and SSVEPExo dataset (:gh:`59` by `Sylvain Chevallier`_)

Bugs
~~~~
- None

API changes
~~~~~~~~~~~
- None



.. _Alexandre Barachant: https://github.com/alexandrebarachant
.. _Quentin Barthelemy: https://github.com/qbarthelemy
.. _Erik Bjäreholt: https://github.com/ErikBjare
.. _Sylvain Chevallier: https://github.com/sylvchev
.. _Lucas Custódio: https://github.com/lucascust
.. _Mohammad Mostafa Farzan: https://github.com/m2-farzan
.. _Ramiro Gatti: https://github.com/ragatti
.. _Vladislav Goncharenko: https://github.com/v-goncharenko
.. _Ali Abdul Hussain: https://github.com/AliAbdulHussain
.. _Vinay Jayaram: https://github.com/vinay-jayaram
.. _Divyesh Narayanan: https://github.com/Div12345
.. _Pedro L. C. Rodrigues: https://github.com/plcrodrigues
.. _Robin Schirrmeister: https://github.com/robintibor
.. _Jan Sosulski: https://github.com/jsosulski
.. _Pierre Guetschel: https://github.com/PierreGtch
.. _Ludovic Darmet: https://github.com/ludovicdmt
