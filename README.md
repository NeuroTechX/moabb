# Mother of all BCI Benchmarks

<p align=center>
  <img alt="banner" src="/images/M.png/">
</p>
<p align=center>
  Build a comprehensive benchmark of popular BCI algorithms applied on an extensive list of freely available EEG datasets.
</p>

## Disclaimer

**This is an open science project that may evolve depending on the need of the
community.**

[![Build Status](https://github.com/NeuroTechX/moabb/workflows/Test/badge.svg)](https://github.com/NeuroTechX/moabb/actions?query=branch%3Amaster)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/moabb?color=blue&style=plastic)](https://img.shields.io/pypi/v/moabb)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/moabb?color=brightgreen&style=plastic)](https://img.shields.io/pypi/dm/moabb)

## Welcome!

First and foremost, Welcome! :tada: Willkommen! :confetti_ball: Bienvenue!
:balloon::balloon::balloon:

Thank you for visiting the Mother of all BCI Benchmark repository.

This document is a hub to give you some information about the project. Jump straight to
one of the sections below, or just scroll down to find out more.

- [What are we doing? (And why?)](#what-are-we-doing)
- [Installation](#installation)
- [Running](#running)
- [Supported datasets](#supported-datasets)
- [Who are we?](#who-are-we)
- [Get in touch](#contact-us)
- [Documentation](#documentation)
- [Architecture and main concepts](#architecture-and-main-concepts)
- [Citing MOABB and related publications](#citing-moabb-and-related-publications)

## What are we doing?

### The problem

- Reproducible Research in BCI has a long way to go.
- While many BCI datasets are made freely available, researchers do not publish code, and
  reproducing results required to benchmark new algorithms turns out to be trickier than
  it should be.
- Performances can be significantly impacted by parameters of the preprocessing steps,
  toolboxes used and implementation “tricks” that are almost never reported in the
  literature.

As a result, there is no comprehensive benchmark of BCI algorithms, and newcomers are
spending a tremendous amount of time browsing literature to find out what algorithm works
best and on which dataset.

### The solution

The Mother of all BCI Benchmarks allows to:

- Build a comprehensive benchmark of popular BCI algorithms applied on an extensive list
  of freely available EEG datasets.
- The code will be made available on github, serving as a reference point for the future
  algorithmic developments.
- Algorithms can be ranked and promoted on a website, providing a clear picture of the
  different solutions available in the field.

This project will be successful when we read in an abstract “ … the proposed method
obtained a score of 89% on the MOABB (Mother of All BCI Benchmarks), outperforming the
state of the art by 5% ...”.

## Installation

### Pip installation

To use MOABB, you could simply do: \
`pip install MOABB` \
See [Troubleshooting](#Troubleshooting) section if you have a problem.

### Manual installation

You could fork or clone the repository and go to the downloaded directory, then run:

1. install `poetry` (only once per machine):\
   `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`\
   or [checkout installation instruction](https://python-poetry.org/docs/#installation) or
   use [conda forge version](https://anaconda.org/conda-forge/poetry)
1. (Optional, skip if not sure) Disable automatic environment creation:\
   `poetry config virtualenvs.create false`
1. install all dependencies in one command (have to be run in the project directory):\
   `poetry install`

See [contributors' guidelines](CONTRIBUTING.md) for detailed explanation.

### Requirements we use

See `pyproject.toml` file for full list of dependencies

## Running

### Verify Installation

To ensure it is running correctly, you can also run

```
python -m unittest moabb.tests
```

once it is installed.

### Use MOABB

First, you could take a look at our [tutorials](./tutorials) that cover the most important
concepts and use cases. Also, we have a several [examples](./examples/) available.

You might be interested in [MOABB documentation][link_moabb_docs]

### Troubleshooting

Currently pip install moabb fails when pip version < 21, e.g. with 20.0.2 due to an `idna`
package conflict. Newer pip versions resolve this conflict automatically. To fix this you
can upgrade your pip version using: `pip install -U pip` before installing `moabb`.

## Supported datasets

The list of supported datasets can be found here :
http://moabb.neurotechx.com/docs/datasets.html

### Submit a new dataset

you can submit a new dataset by mentioning it to this
[issue](https://github.com/NeuroTechX/moabb/issues/1). The datasets currently on our radar
can be seen [here] (https://github.com/NeuroTechX/moabb/wiki/Datasets-Support)

## Who are we?

The founders of the Mother of all BCI Benchmarks are [Alexander Barachant][link_alex_b]
and [Vinay Jayaram][link_vinay]. This project is under the umbrella of
[NeuroTechX][link_neurotechx], the international community for NeuroTech enthusiasts. The
project is currently maintained by [Sylvain Chevallier][link_sylvain].

### What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation
and technical writing and project management.

We'd love your feedback along the way.

Our primary goal is to build a comprehensive benchmark of popular BCI algorithms applied
on an extensive list of freely available EEG datasets, and we're excited to support the
professional development of any and all of our contributors. If you're looking to learn to
code, try out working collaboratively, or translate your skills to the digital domain,
we're here to help.

### Get involved

If you think you can help in any of the areas listed above (and we bet you can) or in any
of the many areas that we haven't yet thought of (and here we're _sure_ you can) then
please check out our [contributors' guidelines](CONTRIBUTING.md) and our
[roadmap](ROADMAP.md).

Please note that it's very important to us that we maintain a positive and supportive
environment for everyone who wants to participate. When you join us we ask that you follow
our [code of conduct](CODE_OF_CONDUCT.md) in all interactions both on and offline.

## Contact us

If you want to report a problem or suggest an enhancement, we'd love for you to
[open an issue](../../issues) at this github repository because then we can get right on
it.

For a less formal discussion or exchanging ideas, you can also reach us on the [Gitter
channel][link_gitter] or join our weekly office hours! This an open video meeting
happening on a [regular basis](https://github.com/NeuroTechX/moabb/issues/191), please ask
the link on the gitter channel. We are also on [NeuroTechX slack #moabb
channel][link_neurotechx_signup].

## Architecture and Main Concepts

<p align="center">
  <img alt="banner" src="/images/architecture.png/" width="400">
</p>
There are 4 main concepts in the MOABB: the datasets, the paradigm, the evaluation, and the pipelines. In addition, we offer statistical and visualization utilities to simplify the workflow.

### Datasets

A dataset handles and abstracts low-level access to the data. The dataset will read data
stored locally, in the format in which they have been downloaded, and will convert them
into a MNE raw object. There are options to pool all the different recording sessions per
subject or to evaluate them separately.

### Paradigm

A paradigm defines how the raw data will be converted to trials ready to be processed by a
decoding algorithm. This is a function of the paradigm used, i.e. in motor imagery one can
have two-class, multi-class, or continuous paradigms; similarly, different preprocessing
is necessary for ERP vs ERD paradigms.

### Evaluations

An evaluation defines how we go from trials per subject and session to a generalization
statistic (AUC score, f-score, accuracy, etc) -- it can be either within-recording-session
accuracy, across-session within-subject accuracy, across-subject accuracy, or other
transfer learning settings.

### Pipelines

Pipeline defines all steps required by an algorithm to obtain predictions. Pipelines are
typically a chain of sklearn compatible transformers and end with a sklearn compatible
estimator. See
[Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
for more info.

### Statistics and visualization

Once an evaluation has been run, the raw results are returned as a DataFrame. This can be
further processed via the following commands to generate some basic visualization and
statistical comparisons:

```
from moabb.analysis import analyze

results = evaluation.process(pipeline_dict)
analyze(results)
```

## Citing MOABB and related publications

To cite MOABB, you could use the following paper:

> Vinay Jayaram and Alexandre Barachant.
> ["MOABB: trustworthy algorithm benchmarking for BCIs."](http://iopscience.iop.org/article/10.1088/1741-2552/aadea0/meta)
> Journal of neural engineering 15.6 (2018): 066011.
> [DOI](https://doi.org/10.1088/1741-2552/aadea0)

If you publish a paper using MOABB, please contact us on [gitter][link_gitter] or open an
issue, and we will add your paper to the
[dedicated wiki page](https://github.com/NeuroTechX/moabb/wiki/MOABB-bibliography).

## Thank You

Thank you so much (Danke schön! Merci beaucoup!) for visiting the project and we do hope
that you'll join us on this amazing journey to build a comprehensive benchmark of popular
BCI algorithms applied on an extensive list of freely available EEG datasets.

[link_alex_b]: http://alexandre.barachant.org/
[link_vinay]: https://ei.is.tuebingen.mpg.de/~vjayaram
[link_neurotechx]: http://neurotechx.com/
[link_sylvain]: https://sylvchev.github.io/
[link_neurotechx_signup]: https://neurotechx.com/
[link_gitter]: https://gitter.im/moabb_dev/community
[link_moabb_docs]: http://moabb.neurotechx.com/docs/index.html
[link_arxiv]: https://arxiv.org/abs/1805.06427
[link_jne]: http://iopscience.iop.org/article/10.1088/1741-2552/aadea0/meta
