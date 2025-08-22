.. _whats_new:

.. currentmodule:: moabb

What's new
==========


.. NOTE: there are 3 separate sections for changes, based on type:

- "Enhancements" for new features
- "Bugs" for bug fixes
- "API changes" for backward-incompatible changes

.. _current:


Develop branch  - 1.4  (dev)
--------------------------------------
Enhancements
~~~~~~~~~~~~
- Adding :class:`moabb.datasets.Kojima2024A` (:gh:`807` by `Simon Kojima`_)

- Add new dataset :class:`moabb.datasets.BNCI2003_IVa` dataset (:gh:`811` by `Griffin Keeler`_)

Bugs
~~~~
- Fixing label swapped issue with  :class:`moabb.datasets.Kalunga2016` dataset (:gh:`814` by `Griffin Keeler`_)

API changes
~~~~~~~~~~~


Develop branch  - 1.3  (Stable - PyPi)
--------------------------------------

Enhancements
~~~~~~~~~~~~
- Adding a tutorial for :class:`moabb.evaluations.splitters.WithinSessionSplitter` (:gh:`776` by `Thomas Kooiman`_, `Paul Verhoeven`_, `Jorge Sanmartin Martinez`_, and `Radovan Vodila`_ )
- Adding new motor imagery dataset, Dreyer2023 (:gh:`404` by `Sara Sedlar`_, `Sylvain Chevallier`_ and `Bruno Aristimunha`_)
- Reordering the examples in the documentation (:gh:`706` by `Bruno Aristimunha`_)
- Creating the meta information for the BIDS converted datasets (:gh:`688` by `Bruno Aristimunha`_)
- Adding :class:`moabb.datasets.Beetl2021_A` and :class:`moabb.datasets.Beetl2021_B` (:gh:`675` by `Samuel Boehm`_)
- Adding :class:`moabb.evaluations.splitters.CrossSubjectSplitter` (:gh:`722` by `Bruna Lopes`_ and `Bruno Aristimunha`_)
- Adding :class:`moabb.evaluations.splitters.CrossSessionSplitter` (:gh:`720` by `Bruna Lopes`_ and `Bruno Aristimunha`_)
- Adding :class:`moabb.datasets.base.BaseBIDSDataset` and :class:`moabb.datasets.base.LocalBIDSDataset` (:gh:`724` by `Pierre Guetschel`_)
- Adding :func:`moabb.analysis.plotting.dataset_bubble_plot` plus the corresponding tutorial (:gh:`753` by `Pierre Guetschel`_)
- Adding :func:`moabb.datasets.utils.plot_all_datasets` and update the tutorial (:gh:`758` by `Pierre Guetschel`_)
- Improve the dataset model cards in each API page (:gh:`765` by `Pierre Guetschel`_)
- Refactor :class:`moabb.evaluation.CrossSessionEvaluation`, :class:`moabb.evaluation.CrossSubjectEvaluation` and  :class:`moabb.evaluation.WithinSessionEvaluation` to use the new splitter classes (:gh:`769` by `Bruno Aristimunha`_)
- Adding tutorial on using mne-features (:gh:`762` by `Alexander de Ranitz`_, `Luuk Neervens`_, `Charlynn van Osch`_ and `Bruno Aristimunha`_)
- Creating tutorial to expose the pre-processing steps (:gh:`771` by `Bruno Aristimunha`_)
- Add function to auto-generate tables for the paper results documentation page (:gh:`785` by `Lucas Heck`_)
- Improving the Filterbank tutorial and implementing the mutual information selection to reproduce the FilterbankCSP (:gh:`787` by `Bruno Aristimunha`_)
- A tutorial on how to create and use a MOABB dataset from X y (non continuous, epoched) data (:gh:`800` by `Anton Andreev`_)
- Improving the parallel writing of results (:gh:`803` by `Bruno Aristimunha`_)

Bugs
~~~~
- Fix regression in evaluations ignoring ``process_pipeline`` flag (:gh:`774` by `Bruno Aristimunha`_)
- Fix caching issue with incomplete results (:gh:`715` by `Sylvain Chevallier`_)
- Fix learning curve example (:gh:`717` by `Pierre Guetschel`_)
- Pick all data channels in filter preprocessing step (:gh:`729` by `Pierre Guetschel`_)
- Fix CI for permutation testing (:gh:`757` by `Quentin Barthelemy`_)
- Fix download issue with Schirrmeister2017 dataset (:gh:`751` by `Zheyu Yao`_)
- Fix code carbon example code (:gh:`777` by `Amar Enkhbat`_)
- Including the fix_bad_channels for the :class:`moabb.datasets.Stieger2021` (:gh:`783` by `Bruno Aristimunha`_)
- Fix the :class:`moabb.datasets.Wang2016` (:gh:`781` by `Ulysse Durand`_)
- Fix warnings raised when building the documentation (:gh:`784` by `Lucas Heck`_)
- Remove an unnecessary line in the README.md (:gh:`791` by `Lionel Kusch`_)
- Update the dead link about the tutorial of GitHub in CONTRIBUTING.md (:gh:`792` by `Lionel Kusch`_)
- Fix: number of trial per class for PHMD_ML dataset (:gh:`797` by `Gregoire Cattan`_)
- Converting the :class:`moabb.datasets.Zhou2016` to BIDS (:gh:`802` by `Bruno Aristimunha`_)

API changes
~~~~~~~~~~~
- Removing the deep learning module from inside moabb in favour of braindecode integration (:gh:`692` by `Bruno Aristimunha`_ )


Version - 1.2.0
----------------


Enhancements
~~~~~~~~~~~~
- Adding :class:`moabb.evaluations.splitters.WithinSessionSplitter` (:gh:`664` by `Bruna Lopes_`)
- Update version of pyRiemann to 0.7 (:gh:`671` by `Gregoire Cattan`_)
- Add columns definitions in the datasets doc (:gh:`672` by `Pierre Guetschel`_)
- Add ERP CORE datasets :class:`moabb.datasets.ErpCore2021` dataset (:gh:`627` by `Taha Habib`_)
- Update paths of BIDS cache to better follow the standards. Cache created in previous MOABB versions should still be compatible (:gh:`707` by `Pierre Guetschel`_)

Bugs
~~~~

- Fix Stieger2021 dataset bugs (:gh:`651` by `Martin Wimpff`_)
- Unpinning major version Scikit-learn and numpy (:gh:`652` by `Bruno Aristimunha`_)
- Replacing the func:`numpy.string_` to func:`numpy.bytes_` (:gh:`665` by `Bruno Aristimunha`_)
- Fixing the set_download_dir that was not working when we tried to set the dir more than 10 times at the same time (:gh:`668` by `Bruno Aristimunha`_)
- Creating stimulus channels in :class:`moabb.datasets.Zhou2016` and :class:`moabb.datasets.PhysionetMI` to allow braindecode compatibility (:gh:`669` by `Bruno Aristimunha`_)
- Improving the CI (:gh:`686` by `Bruno Aristimunha`_)
- Making the download test work again (:gh:`693` by `Bruno Aristimunha`_)
- Fix the EpochSelectChannel that caused incorrect channel selection in `example <examples/plot_Hinss2021_classification.py>`__ (:gh:`685` by `AFF`)
- Fixing the logger on the Stieger2021 and Wang2016 dataset (:gh:`693` by `Bruno Aristimunha`_)
- Change the way of creating the path to the folder (:gh:`697` by `Sebastien Velut`_)
- Fixing bug with braindecode and moabb datasets EPFLP300 (:gh:`696` by `Bruno Aristimunha`_)
- Fixing the dataset details for bids conversion (:gh:`698` by `Bruno Aristimunha`_)
- Fixing unit issue and lack of montage with :class:`moabb.datasets.Rodrigues2017`, :class:`moabb.datasets.Rodrigues2017`, :class:`moabb.datasets.BaseCastillos2023`,  :class:`moabb.datasets.BaseCastillos2023`,  :class:`moabb.datasets.Huebner2018`,  :class:`moabb.datasets.Cattan2019_PHMD`, :class:`moabb.datasets.Ofner2017`  (:gh:`700`  `Bruno Aristimunha`_)
- Fix t-test permutation tests (:gh:`684` and :gh:`709` by `Gregoire Cattan`_, `Anton Andreev`_, `Marco Congedo`_ and `Bruno Aristimunha`_)


API changes
~~~~~~~~~~~
- Removing the braindecode module from inside moabb (:gh:`666` by `Bruno Aristimunha`_ )



Version - 1.1.1
----------------

Enhancements
~~~~~~~~~~~~
- Add possibility to use OptunaGridSearch (:gh:`630` by `Igor Carrara`_)
- Add scripts to upload results on PapersWithCode (:gh:`561` by `Pierre Guetschel`_)
- Centralize dataset summary tables in CSV files (:gh:`635` by `Pierre Guetschel`_)
- Add new dataset :class:`moabb.datasets.Liu2024` dataset (:gh:`619` by `Taha Habib`_)
- Add choice to choose the size of time window (by `Sebastien Velut`_)


Bugs
~~~~
- Fix caching in the workflows (:gh:`632` by `Pierre Guetschel`_)

API changes
~~~~~~~~~~~
- Include optuna as soft-dependency in the benchmark function and in the base of evaluation (:gh:`630` by `Igor Carrara`_)



Version - 1.1.0
----------------


Enhancements
~~~~~~~~~~~~

- Add cache option to the evaluation (:gh:`518` by `Bruno Aristimunha`_)
- Option to interpolate channel in paradigms' `match_all` method (:gh:`480` by `Gregoire Cattan`_)
- Add leave k-Subjects out evaluations (:gh:`470` by `Bruno Aristimunha`_)
- Update Braindecode dependency to 0.8 (:gh:`542` by `Pierre Guetschel`_)
- Improve transform function of AugmentedDataset (:gh:`541` by `Quentin Barthelemy`_)
- Add new paper results website (:gh:`556` by `Bruno Aristimunha`_)
- Move cVEP common functions to :mod:`moabb.datasets.utils` (:gh:`564` :gh:`557` by `Pierre Guetschel`_)
- Normalize c-VEP description tables (:gh:`562` :gh:`566` by `Pierre Guetschel`_ and `Bruno Aristimunha`_)
- Update citation in README (:gh:`573` by `Igor Carrara`_)
- Update pyRiemann dependency (:gh:`577` by `Gregoire Cattan`_)
- Add resting stage Hinss2021 dataset (:gh:`580` by `Gregoire Cattan`_ and `Yash Chauhan`_)
- Expose the `learning` rate parameter in the keras deep learning methods and optimize parameters (:gh:`589` and :gh:`592` by `Bruno Aristimunha`_)
- Updating the braindecode pipelines for the new braindecode version 0.8.1 (:gh:`589` by `Bruno Aristimunha`_)
- Add SSVEP and ERP paradigms to DL pipelines (:gh:`590` by `Pierre Guetschel`_)
- Allow to pass a single pipeline file to ``benchmark`` (:gh:`591` by `Pierre Guetschel`_)
- Add new dataset :class:`moabb.datasets.Stieger2021` (:gh:`604` by `Reinmar Kobler`_ and `Bruno Aristimunha`_)
- Exposing the `drop_rate` for all the deep learning parameters (:gh:`592` by `Bruno Aristimunha`_)
- Add new dataset :class:`moabb.datasets.Rodrigues2017` dataset (:gh:`602` by `Gregoire Cattan`_ and `Pedro L. C. Rodrigues`_)
- Change unittest to pytest (:gh:`618` by `Bruno Aristimunha`_)
- Remove tensorflow import warning (:gh:`622` by `Bruno Aristimunha`_)

Bugs
~~~~

- Fix TRCA implementation for different stimulation freqs and for signal filtering (:gh:522 by `Sylvain Chevallier`_)
- Fix saving to BIDS runs with a description string in their name (:gh:`530` by `Pierre Guetschel`_)
- Fix import of keras BatchNormalization for TF 2.13 and higher (:gh:`544` by `Brian Irvine`_)
- Fix the doc summary tables of :class:`moabb.datasets.Lee2019_SSVEP` (:gh:`548` :gh:`547` :gh:`546` by `Pierre Guetschel`_)
- Fix the doc summary for Castillos2023 dataset (:gh:`561` by `Bruno Aristimunha`_)
- Fix format string receiving incorrect number of args in bids interface (:gh:`563` by `Pierre Guetschel`_)
- Fix number of sessions in doc of :class:`moabb.datasets.Sosulski2019` (:gh:`565` by `Pierre Guetschel`_)
- Fix `code` column of :class:`moabb.datasets.CastillosCVEP100` and :class:`moabb.datasets.CastillosCVEP100` (:gh:`567` by `Pierre Guetschel`_)
- MAINT updating the packages pre-release (:gh:`578` by `Bruno Aristimunha`_)
- Fix mne_bids version incompatibility with mne (:gh:`586` by `Bruna Lopes`_)
- Updating the parameters of the SSVEP_TRCA method (:gh:`589` by `Bruno Aristimunha`_)
- Fix and updating the parameters for the benchmark function (:gh:`588` by `Bruno Aristimunha`_)
- Fix result table display (:gh:`599` by `Sylvain Chevallier`_)
- Fix :class:`moabb.datasets.preprocessing.SetRawAnnotations` setting incorrect annotations when the dataset's interval does not start at 0 (:gh:`607` by `Pierre Guetschel`_)
- Fix download link for GigaDB Cho2017 and Lee2019 datasets (:gh:`621` by `Anton Andreev`_)


API changes
~~~~~~~~~~~

- None


Version - 1.0.0
----------------

Enhancements
~~~~~~~~~~~~

- Adding extra thank you section in the documentation (:gh:`390` by `Bruno Aristimunha`_)
- Adding new script to get the meta information of the datasets (:gh:`389` by `Bruno Aristimunha`_)
- Fixing the dataset description based on the meta information (:gh:`389` and `398` by `Bruno Aristimunha`_ and `Sara Sedlar`_)
- Adding second deployment of the documentation (:gh:`374` by `Bruno Aristimunha`_)
- Adding Parallel evaluation for :func:`moabb.evaluations.WithinSessionEvaluation` , :func:`moabb.evaluations.CrossSessionEvaluation` (:gh:`364` by `Bruno Aristimunha`_)
- Add example with VirtualReality BrainInvaders dataset (:gh:`393` by `Gregoire Cattan`_ and `Pedro L. C. Rodrigues`_)
- Adding saving option for the models (:gh:`401` by `Bruno Aristimunha`_ and `Igor Carrara`_)
- Adding example to load different type of models (:gh:`401` by `Bruno Aristimunha`_ and `Igor Carrara`_)
- Add resting state paradigm with dataset and example (:gh:`400` by `Gregoire Cattan`_ and `Pedro L. C. Rodrigues`_)
- Speeding the augmentation method by 400% with NumPy vectorization  (:gh:`419` by `Bruno Aristimunha`_)
- Add possibility to convert datasets to BIDS, plus `example <examples/example_bids_conversion.py>`__ (PR :gh:`408`, PR :gh:`391` by `Pierre Guetschel`_ and `Bruno Aristimunha`_)
- Allow caching intermediate processing steps on disk, plus `example <examples/example_disk_cache.py>`__ (PR :gh:`408`, issue :gh:`385` by `Pierre Guetschel`_)
- Restructure the paradigms and datasets to move all preprocessing steps to :mod:`moabb.datasets.preprocessing` and as sklearn pipelines (PR :gh:`408` by `Pierre Guetschel`_)
- Add :func:`moabb.paradigms.FixedIntervalWindowsProcessing` and :func:`moabb.paradigms.FilterBankFixedIntervalWindowsProcessing`, plus `example <examples/example_fixed_interval_windows.py>`__ (PR :gh:`408`, issue :gh:`424` by `Pierre Guetschel`_)
- Define :func:`moabb.paradigms.base.BaseProcessing`, common parent to :func:`moabb.paradigms.base.BaseParadigm` and :func:`moabb.paradigms.BaseFixedIntervalWindowsProcessing` (PR :gh:`408` by `Pierre Guetschel`_)
- Allow passing a fixed processing pipeline to :func:`moabb.paradigms.base.BaseProcessing.get_data` and cache its result on disk (PR :gh:`408`, issue :gh:`367` by `Pierre Guetschel`_)
- Update :func:`moabb.datasets.fake.FakeDataset`'s code to be unique for each parameter combination (PR :gh:`408` by `Pierre Guetschel`_)
- Systematically set the annotations when loading data, eventually using the stim channel (PR :gh:`408` by `Pierre Guetschel`_)
- Allow :func:`moabb.datasets.utils.dataset_search` to search across paradigms ``paradigm=None`` (PR :gh:`408` by `Pierre Guetschel`_)
- Improving the review processing with more pre-commit bots (:gh:`435` by `Bruno Aristimunha`_)
- Add methods ``make_processing_pipelines`` and ``make_labels_pipeline`` to :class:`moabb.paradigms.base.BaseProcessing` (:gh:`447` by `Pierre Guetschel`_)
- Pipelines' digests are now computed from the whole processing+classification pipeline (:gh:`447` by `Pierre Guetschel`_)
- Update all dataset codes to remove white spaces and underscores (:gh:`448` by `Pierre Guetschel`_)
- Add :func:`moabb.utils.depreciated_alias` decorator (:gh:`455` by `Pierre Guetschel`_)
- Rename many dataset class names to standardize and deprecate old names (:gh:`455` by `Pierre Guetschel`_)
- Change many dataset codes to match the class names (:gh:`455` by `Pierre Guetschel`_)
- Add :obj:`moabb.datasets.compound_dataset.utils.compound_dataset_list`  (:gh:`455` by `Pierre Guetschel`_)
- Add c-VEP paradigm and Thielen2021 c-VEP dataset (:gh:`463` by `Jordy Thielen`_)
- Add option to plot scores vertically. (:gh:`417` by `Sara Sedlar`_)
- Change naming scheme for runs and sessions to align to BIDS standard (:gh:`471` by `Pierre Guetschel`_)
- Increase the python version to 3.11 (:gh:`470` by `Bruno Aristimunha`_)
- Add match_all method in paradigm to support CompoundDataset evaluation with MNE epochs (:gh:`473` by `Gregoire Cattan`_)
- Automate setting of event_id in compound dataset and add `data_origin` information to the data (:gh:`475` by `Gregoire Cattan`_)
- Add possibility of not saving the model (:gh:`489` by `Igor Carrara`_)
- Add CVEP and BurstVEP dataset from Castillos from Toulouse lab (:gh:`531` by `Sebastien Velut`_)
- Add c-VEP dataset from Thielen et al. 2015 (:gh:`557` by `Jordy Thielen`_)

Bugs
~~~~

- Restore 3 subject from Cho2017 (:gh:`392` by `Igor Carrara`_ and `Sylvain Chevallier`_)
- Correct downloading with VirtualReality BrainInvaders dataset (:gh:`393` by `Gregoire Cattan`_)
- Rename event `subtraction` in :func:`moabb.datasets.Shin2017B` (:gh:`397` by `Pierre Guetschel`_)
- Save parameters of :func:`moabb.datasets.PhysionetMI` (:gh:`403` by `Pierre Guetschel`_)
- Fixing issue with parallel evaluation (:gh:`401` by `Bruno Aristimunha`_ and `Igor Carrara`_)
- Fixing SSLError from BCI competition IV (:gh:`404` by `Bruno Aristimunha`_)
- Fixing :func:`moabb.datasets.bnci.MNEBNCI.data_path` that returned the data itself instead of paths (:gh:`412` by `Pierre Guetschel`_)
- Adding :func:`moabb.datasets.fake` in the init file to use in braindecode object (:gh:`414` by `Bruno Aristimunha`_)
- Fixing the parallel download issue when the dataset have the same directory (:gh:`421` by `Sara Sedlar`_)
- Fixing fixes the problem with the annotation loading for the P300 datasets Sosulski2019, Huebner2017 and Huebner2018 (:gh:`396` by `Sara Sedlar`_)
- Removing the print in the dataset list (:gh:`423` by `Bruno Aristimunha`_)
- Fixing bug in :func:`moabb.pipeline.utils_pytorch.BraindecodeDatasetLoader` where incorrect y was used in transform calls (:gh:`426` by `Gabriel Schwartz`_)
- Fixing one test in :func:`moabb.pipeline.utils_pytorch.BraindecodeDatasetLoader` (:gh:`426` by `Bruno Aristimunha`_)
- Fix :func:`moabb.benchmark` overwriting ``include_datasets`` list (:gh:`408` by `Pierre Guetschel`_)
- Fix :func:`moabb.paradigms.base.BaseParadigm` using attributes before defining them  (PR :gh:`408`, issue :gh:`425` by `Pierre Guetschel`_)
- Fix :func:`moabb.paradigms.FakeImageryParadigm`, :func:`moabb.paradigms.FakeP300Paradigm` and :func:`moabb.paradigms.FakeSSVEPParadigm` ``is_valid`` methods to only accept the correct datasets (PR :gh:`408` by `Pierre Guetschel`_)
- Fix ``dataset_list`` construction, which could be empty due to bad import order (PR :gh:`449` by `Thomas Moreau`_).
- Fixing dataset downloader from servers with non-http (PR :gh:`433` by `Sara Sedlar`_)
- Fix ``dataset_list`` to include deprecated datasets (PR :gh:`464` by `Bruno Aristimunha`_)
- Fixed bug in :func:`moabb.analysis.results.get_string_rep` to handle addresses such as 0x__0A as well (PR :gh:`468` by `Anton Andreev`_)
- Moving the :func:`moabb.evualation.grid_search` to inside the base evaluation (:gh:`487` by `Bruno Aristimunha`_)
- Removing joblib Parallel (:gh:`488` by `Igor Carrara`_)
- Fix case when events specified via ``raw.annotations`` but no events (:gh:`491` by `Pierre Guetschel`_)
- Fix bug in downloading Shin2017A dataset (:gh:`493` by `Igor Carrara`_)
- Fix the cropped option in the dataset preprocessing (:gh:`502` by `Bruno Aristimunha`_)
- Fix bug in :func:`moabb.datasets.utils.dataset_search` with missing cvep paradigm (:gh:`557` by `Jordy Thielen`_)
- Fix mistakes in :func:`moabb.datasets.thielen2021` considering wrong docs and hardcoded trial stim channel (:gh:`557` by `Jordy Thielen`_)

API changes
~~~~~~~~~~~

- None


Version - 0.5.0
---------------

Enhancements
~~~~~~~~~~~~
- Speeding the augmentation model (:gh:`365` by `Bruno Aristimunha`_)
- Add VirtualReality BrainInvaders dataset (:gh:`358` by `Gregoire Cattan`_)
- Switch to python-3.8, update dependencies, fix code link in doc, add `code coverage <https://app.codecov.io/gh/NeuroTechX/moabb>`__ (:gh:`315` by `Sylvain Chevallier`_)
- Adding a comprehensive benchmarking function (:gh:`264` by `Divyesh Narayanan`_ and `Sylvain Chevallier`_)
- Add meta-information for datasets in documentation (:gh:`317` by `Bruno Aristimunha`_)
- Add GridSearchCV for different evaluation procedure (:gh:`319` by `Igor Carrara`_)
- Add new tutorial to benchmark with GridSearchCV (:gh:`323` by `Igor Carrara`_)
- Add six deep learning models (Tensorflow), and build a tutorial to show to use the deep learning model (:gh:`326` by `Igor Carrara`_, `Bruno Aristimunha`_ and `Sylvain Chevallier`_)
- Add a augmentation model to the pipeline (:gh:`326` by `Igor Carrara`_)
- Add BrainDecode example (:gh:`340` by `Igor Carrara`_ and `Bruno Aristimunha`_)
- Add Google Analytics to the documentation (:gh:`335` by `Bruno Aristimunha`_)
- Add support to Braindecode classifier (:gh:`328` by `Bruno Aristimunha`_)
- Add CodeCarbon to track emission CO₂ (:gh:`350` by `Igor Carrara`_, `Bruno Aristimunha`_ and `Sylvain Chevallier`_)
- Add CodeCarbon example (:gh:`356` by `Igor Carrara`_ and `Bruno Aristimunha`_)
- Add MsetCCA method for SSVEP classification, parametrise CCA `n_components` in CCA based methods (:gh:`359` by `Emmanuel Kalunga`_ and `Sylvain Chevallier`_)
- Set epochs' `metadata` field in `get_data` (:gh:`371` by `Pierre Guetschel`_)
- Add possibility to use transformers to apply fixed pre-processings before evaluations (:gh:`372` by `Pierre Guetschel`_)
- Add `seed` parameter to `FakeDataset` (:gh:`372` by `Pierre Guetschel`_)

Bugs
~~~~
- Fix circular import with braindecode (:gh:`363` by `Bruno Aristimunha`_)
- Fix bug for MotorImagery when we handle all events (:gh:`327` by `Igor Carrara`_)
- Fixing CI to handle with new deep learning dependencies (:gh:`332` and :gh:`326` by `Igor Carrara`_, `Bruno Aristimunha`_ and `Sylvain Chevallier`_)
- Correct CI error due to isort (:gh:`330` by `Bruno Aristimunha`_)
- Restricting Python <= 3.11 version and adding tensorflow, keras, scikeras, braindecode, skorch and torch, as optional dependence (:gh:`329` by `Bruno Aristimunha`_)
- Fix numpy variable to handle with the new version of python (:gh:`324` by `Bruno Aristimunha`_)
- Correct CI error due to black (:gh:`292` by `Sylvain Chevallier`_)
- Preload Schirrmeister2017 raw files (:gh:`290` by `Pierre Guetschel`_)
- Incorrect event assignation for Lee2019 in MNE >= 1.0.0 (:gh:`298` by `Sylvain Chevallier`_)
- Correct usage of name simplification function in analyze (:gh:`306` by `Divyesh Narayanan`_)
- Fix downloading path issue for Weibo2014 and Zhou2016, numy error in DemonsP300 (:gh:`315` by `Sylvain Chevallier`_)
- Fix unzip error for Huebner2017 and Huebner2018 (:gh:`318` by `Sylvain Chevallier`_)
- Fix n_classes when events set to None (:gh:`337` by `Igor Carrara`_ and `Sylvain Chevallier`_)
- Change n_jobs=-1 to self.n_jobs in GridSearch (:gh:`344` by `Igor Carrara`_)
- Fix dropped epochs issue (:gh:`371` by `Pierre Guetschel`_)
- Fix redundancy website issue (:gh:`372` by `Bruno Aristimunha`_)

API changes
~~~~~~~~~~~

- None

Version - 0.4.6
---------------

Enhancements
~~~~~~~~~~~~

- Add P300 BrainInvaders datasets (:gh:`283` by `Sylvain Chevallier`_)
- Add explicit warning when lambda function are used to parametrize pipelines (:gh:`278` by `Jan Sosulski`_)


Bugs
~~~~

- Correct default path for ERP visualization (:gh:`279` by `Jan Sosulski`_)
- Correct documentation (:gh:`282` and :gh:`284` by `Jan Sosulski`_)



Version - 0.4.5
---------------

Enhancements
~~~~~~~~~~~~

- Progress bars, pooch, tqdm (:gh:`258` by `Divyesh Narayanan`_ and `Sylvain Chevallier`_)
- Adding test and example for set_download_dir (:gh:`249` by `Divyesh Narayanan`_)
- Update to newer version of Schirrmeister2017 dataset (:gh:`265` by `Robin Schirrmeister`_)
- Adding Huebner2017 and Huebner2018 P300 datasets (:gh:`260`  by `Jan Sosulski`_)
- Adding Sosulski2019 auditory P300 datasets (:gh:`266`  by `Jan Sosulski`_)
- New script to visualize ERP on all datasets, as a sanity check (:gh:`261`  by `Jan Sosulski`_)

Bugs
~~~~

- Removing dependency on mne method for PhysionetMI data downloading, renaming runs (:gh:`257` by `Divyesh Narayanan`_)
- Correcting events management in Schirrmeister2017, renaming session and run (:gh:`255` by `Pierre Guetschel`_ and `Sylvain Chevallier`_)
- Switch session and runs in MAMEM1, 2 and 3 to avoid error in WithinSessionEvaluation (:gh:`256` by `Sylvain Chevallier`_)
- Correct doctstrings for the documentation, including Lee2017 (:gh:`256` by `Sylvain Chevallier`_)


Version - 0.4.4
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
- Correct failing CI tests (:gh:`100` by `Sylvain Chevallier`_)
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

.. _Zheyu Yao: https://github.com/zyao197
.. _Martin Wimpff: https://github.com/martinwimpff
.. _Reinmar Kobler: https://github.com/rkobler
.. _Gabriel Schwartz: https://github.com/Kaos9001
.. _Sara Sedlar: https://github.com/Sara04
.. _Emmanuel Kalunga: https://github.com/emmanuelkalunga
.. _Gregoire Cattan: https://github.com/gcattan
.. _Anton Andreev: https://github.com/toncho11
.. _Igor Carrara: https://github.com/carraraig
.. _Bruno Aristimunha: https://github.com/bruAristimunha
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
.. _Thomas Moreau: https://github.com/tommoral
.. _Jordy Thielen: https://github.com/thijor
.. _Sebastien Velut: https://github.com/swetbear
.. _Brian Irvine: https://github.com/brianjohannes
.. _Bruna Lopes: https://github.com/brunaafl
.. _Yash Chauhan: https://github.com/jiggychauhi
.. _Taha Habib: https://github.com/tahatt13
.. _AFF: https://github.com/allwaysFindFood
.. _Marco Congedo: https://github.com/Marco-Congedo
.. _Samuel Boehm: https://github.com/Samuel-Boehm
.. _Amar Enkhbat: https://github.com/amar-enkhbat
.. _Alexander de Ranitz: https://github.com/alexander-de-ranitz
.. _Luuk Neervens: https://github.com/LuukNeervens
.. _Charlynn van Osch: https://github.com/charlynnvanosch
.. _Paul Verhoeven: https://github.com/PaulusBoskabouter
.. _Thomas Kooiman: https://github.com/jellymace
.. _Jorge Sanmartin Martinez: https://github.com/jorgesanmar
.. _Radovan Vodila: https://github.com/rvodila
.. _Ulysse Durand: https://github.com/UlysseDurand
.. _Lucas Heck: https://github.com/lucas-heck
.. _Simon Kojima: https://github.com/simonkojima
.. _Griffin Keeler: https://github.com/griffinkeeler
.. _ Kosei Nakada: https://github.com/ponpopon
