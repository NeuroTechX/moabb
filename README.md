# Mother of all BCI Benchmark

Reproducible Research in BCI has a long way to go. While many BCI datasets are made freely available, researchers do not publish code, and reproducing results required to benchmark new algorithms turns out to be more tricky than it should be. Performances can be significantly impacted by parameters of the preprocessing steps, toolboxes used and implementation “tricks” that are almost never reported in the literature. As a results, there is no comprehensive benchmark of BCI algorithm, and newcomers are spending a tremendous amount of time browsing literature to find out what algorithm works best and on which dataset.

The Goal of this project is to build a comprehensive benchmark of popular BCI algorithms applied on an extensive list of freely available EEG dataset. The code will be made available on github, serving as a reference point for the future algorithmic developments. Algorithms can be ranked and promoted on a website, providing a clear picture of the different solutions available in the field.

This project will be successful when we read in an abstract “ … the proposed method obtained a score of 89% on the MOABB (Mother of All BCI Benchmark), outperforming the state of the art by 5%  ...”

# Disclaimer

**This is work in progress. API will change significantly (as well as the results of the benchmark).**

# Install

`python setup.py develop`

# requirements

Many of the datasets are currently made available through an ongoing MNE PR : https://github.com/mne-tools/mne-python/pull/4019

to have access to the datasets, please checkout the corresponding branch.
https://help.github.com/articles/checking-out-pull-requests-locally/


# datasets

Currently, there is 9 motor Imagery dataset supported. 5 of them will be automatically downloaded through MNE.
The other 4 need to be downloaded manually :

- *Alex_mi* : This dataset is not released yet. it should be done soon. contact alexandre in the meantime.
- *OpenvibeMI* : can be downloaded [here](http://openvibe.inria.fr/datasets-downloads/)
- *gigadb_mi* : can be downloaded [here](ftp://climb.genomics.cn/pub/10.5524/100001_101000/100295/mat_data/)
- *bbci_eeg_fnirs* : can be downloaded [here](http://doc.ml.tu-berlin.de/hBCI/)

you can submit new dataset by filling this [form](https://docs.google.com/forms/d/e/1FAIpQLScxbpqK4omKsUs4tA2XpfeHJATo_SbYvT0hpxoeKDb5k_TZvQ/viewform). Please check first that the algorithm is not in the [list](https://docs.google.com/spreadsheets/d/1fQNFXGu1J1yJ9jFCer9EQQatjCPJWg7O-uCGF0Z4PiM/edit).  

# Architecture and main concepts

there is 3 main concepts in the MOABB: the datasets, the context and the pipelines.

### datasets

A dataset handle and abstract low level access to the data. the dataset will takes data stored locally, in the format in which they have been downloaded, and will convert them into a MNE raw object.

### contexts

A context define how the raw data will be converted to trials ready to be processed by a decoding algorithm. The context also define how performances are evaluates, i.e. define the cross-validation procedure and the metric used.
A single dataset can lead to multiple context. For example, a multi-class dataset can be evaluated as a multi-class problem, or as multiples binary classification problem. Similarly, we can have a within subject evaluation context or a cross-subject evaluation.

### pipelines

Pipeline defines all steps required by an algorithm to obtain predictions. Pipelines are typically a chain of sklearn compatible transformers and end with an sklearn compatible estimator.
See [Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for more info.
