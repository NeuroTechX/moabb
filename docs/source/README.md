.. currentmodule:: moabb

# <p align="center">Mother of all BCI Benchmarks</p>

<p align="center">
  <img src="_static/moabb_logo.svg" width="400" height="400" style="display: block; margin: auto;" />
  Build a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive list of freely available EEG datasets.
</p>

## Disclaimer

**This is an open science project that may evolve depending on the need of the
community.**

[![Build Status](https://github.com/NeuroTechX/moabb/workflows/Test/badge.svg)](https://github.com/NeuroTechX/moabb/actions?query=branch%3Amaster)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/moabb?color=blue&style=plastic)](https://img.shields.io/pypi/v/moabb)
[![Downloads](https://pepy.tech/badge/moabb)](https://pepy.tech/project/moabb)

## Welcome!

Thank you for visiting the Mother of all BCI Benchmark documentation and associated
[GitHub repository](https://github.com/NeuroTechX/moabb)

This document is a hub to give you some information about the project. Jump straight to
one of the sections below, or just scroll down to find out more.

### The problem

[Brain-Computer Interfaces](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface)
allow to interact with a computer using brain signals. In this project, we focus mostly on
electroencephalographic signals
([EEG](https://en.wikipedia.org/wiki/Electroencephalography)), that is a very active
research domain, with worldwide scientific contributions. Still:

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
- The code is available on GitHub, serving as a reference point for the future algorithmic
  developments.
- Algorithms can be ranked and promoted on a website, providing a clear picture of the
  different solutions available in the field.

This project will be successful when we read in an abstract “ … the proposed method
obtained a score of 89% on the MOABB (Mother of All BCI Benchmarks), outperforming the
state of the art by 5% ...”.

### Use MOABB

First, you could take a look at our [tutorials](./auto_tutorials/index.html) that cover
the most important concepts and use cases. Also, we have a gallery of
[examples](./auto_examples/index.html) available.

### Troubleshooting

Currently pip install moabb fails when pip version < 21, e.g. with 20.0.2 due to an `idna`
package conflict. Newer pip versions resolve this conflict automatically. To fix this you
can upgrade your pip version using: `pip install -U pip` before installing `moabb`.

## Team

The founders of the Mother of all BCI Benchmarks are [Alexander Barachant][link_alex_b]
and [Vinay Jayaram][link_vinay]. This project is under the umbrella of
[NeuroTechX][link_neurotechx], the international community for NeuroTech enthusiasts. The
project is currently maintained by [Sylvain Chevallier][link_sylvain], [Bruno
Aristimunha][link_bruno], [Igor Carrara][link_igor].

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

## Contact us

If you want to report a problem or suggest an enhancement, we'd love for you to
[open an issue](https://github.com/NeuroTechX/moabb/issues) at this GitHub repository
because then we can get right on it.

For a less formal discussion or exchanging ideas, you can also reach us on the [Gitter
channel][link_gitter] or join our weekly office hours! This an open video meeting
happening on a [regular basis](https://github.com/NeuroTechX/moabb/issues/191), please ask
the link on the gitter channel. We are also on NeuroTechX Slack channel
[#moabb][link_neurotechx_signup].

## Thank You!

Thank you so much (Danke schön! Merci beaucoup!) for visiting the project and we do hope
that you'll join us on this amazing journey to build a comprehensive benchmark of popular
BCI algorithms applied on an extensive list of freely available EEG datasets.

[link_alex_b]: http://alexandre.barachant.org/
[link_vinay]: https://ei.is.tuebingen.mpg.de/~vjayaram
[link_neurotechx]: http://neurotechx.com/
[link_sylvain]: https://sylvchev.github.io/
[link_bruno]: https://www.linkedin.com/in/bruaristimunha/
[link_igor]: https://www.linkedin.com/in/carraraig/
[link_neurotechx_signup]: https://neurotechx.com/
[link_gitter]: https://app.gitter.im/#/room/#moabb_dev_community:gitter.im
[link_moabb_docs]: https://neurotechx.github.io/moabb/
[link_arxiv]: https://arxiv.org/abs/1805.06427
[link_jne]: http://iopscience.iop.org/article/10.1088/1741-2552/aadea0/meta
