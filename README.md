# Mother of all BCI Benchmark

Reproducible Research in BCI has a long way to go. While many BCI datasets are made freely available, researchers do not publish code, and reproducing results required to benchmark new algorithms turns out to be more tricky than it should be. Performances can be significantly impacted by parameters of the preprocessing steps, toolboxes used and implementation “tricks” that are almost never reported in the literature. As a results, there is no comprehensive benchmark of BCI algorithm, and newcomers are spending a tremendous amount of time browsing literature to find out what algorithm works best and on which dataset.

The Goal of this project is to build a comprehensive benchmark of popular BCI algorithms applied on an extensive list of freely available EEG dataset. The code will be made available on github, serving as a reference point for the future algorithmic developments. Algorithms can be ranked and promoted on a website, providing a clear picture of the different solutions available in the field.

This project will be successful when we read in an abstract “ … the proposed method obtained a score of 89% on the MOABB (Mother of All BCI Benchmark), outperforming the state of the art by 5%  ...”

# Disclaimer

**This is work in progress. API will change significantly (as well as the results of the benchmark).**

# Install

`python setup.py develop`

# requirement

Many of the datasets are currently made available through an ongoing MNE PR : https://github.com/mne-tools/mne-python/pull/4019

to have access to the datasets, please checkout the corresponding branch.
https://help.github.com/articles/checking-out-pull-requests-locally/
