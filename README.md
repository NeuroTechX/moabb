# Mother of all BCI Benchmark

Reproducible Research in BCI has a long way to go. While many BCI datasets are made freely available, researchers do not publish code, and reproducing results required to benchmark new algorithms turns out to be more tricky than it should be. Performances can be significantly impacted by parameters of the preprocessing steps, toolboxes used and implementation “tricks” that are almost never reported in the literature. As a results, there is no comprehensive benchmark of BCI algorithm, and newcomers are spending a tremendous amount of time browsing literature to find out what algorithm works best and on which dataset.

The Goal of this project is to build a comprehensive benchmark of popular BCI algorithms applied on an extensive list of freely available EEG dataset. The code will be made available on github, serving as a reference point for the future algorithmic developments. Algorithms can be ranked and promoted on a website, providing a clear picture of the different solutions available in the field.

This project will be successful when we read in an abstract “ … the proposed method obtained a score of 89% on the MOABB (Mother of All BCI Benchmark), outperforming the state of the art by 5%  ...”

# Disclaimer

**This is work in progress. API will change significantly (as well as the results of the benchmark).**

# Install

`python setup.py develop`

# requirements

mne
numpy
scipy
scikit-learn
matplotlib
seaborn
pandas
pyriemann
h5py

# supported datasets

Currently, there are 9 motor Imagery dataset supported, and all of them can be downloaded automatically through the MOABB interface. For more information on the ones not available through the BNCI Horizons 2020 project, see below:

- *Alex_mi* : can be downloaded [here](https://zenodo.org/record/806023)
- *OpenvibeMI* : can be downloaded [here](http://openvibe.inria.fr/datasets-downloads/)
- *gigadb_mi* : can be downloaded [here](ftp://climb.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/)
- *bbci_eeg_fnirs* : can be downloaded [here](http://doc.ml.tu-berlin.de/hBCI/)

### Submit a new dataset

you can submit new dataset by filling this [form](https://docs.google.com/forms/d/e/1FAIpQLScxbpqK4omKsUs4tA2XpfeHJATo_SbYvT0hpxoeKDb5k_TZvQ/viewform). Please check first that the algorithm is not in the [list](https://docs.google.com/spreadsheets/d/1fQNFXGu1J1yJ9jFCer9EQQatjCPJWg7O-uCGF0Z4PiM/edit).  

# Architecture and main concepts

there is 4 main concepts in the MOABB: the datasets, the context, the evaluation, and the pipelines.

### datasets

A dataset handle and abstract low level access to the data. the dataset will
takes data stored locally, in the format in which they have been downloaded, and
will convert them into a MNE raw object. There are options to pool all the
different recording sessions per subject or to evaluate them separately. 

### paradigm

A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm. This is a function of the paradigm used,
i.e. in motor imagery one can have two-class, multi-class, or continuous
paradigms; similarly, different preprocessing is necessary for ERP vs ERD paradigms.

### evaluations

An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) -- it can be either
within-recording-session accuracy, across-session within-subject accuracy,
across-subject accuracy, or other transfer learning settings.

### pipelines

Pipeline defines all steps required by an algorithm to obtain predictions. Pipelines are typically a chain of sklearn compatible transformers and end with an sklearn compatible estimator.
See [Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for more info.

# Installation

To install, fork or clone the repository and go to the downloaded directory,
then run

```
pip install -r requirements.txt
python setup.py develop    # because no stable release yet
```

To ensure it is running correctly, you can also run

```
python -m unittest moabb.tests
```
once it is installed

# How to contribute

1. Look for open issues or open one.
2. Discuss the problem and or propose a solution.
3. Fork this repository and implement the solution.
4. Create a pull request, iterate until it is merged.
