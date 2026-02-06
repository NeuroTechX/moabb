<div align="center" class="moabb-readme-header">
  <img
    src="https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/_static/moabb_notext.svg"
    width="220"
    height="220"
    alt="MOABB logo"
  />
  <h1>Mother of all BCI Benchmarks (MOABB)</h1>
  <p>
    Build a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive list
    of freely available EEG datasets.
  </p>
  <p>
    <a href="https://moabb.neurotechx.com/">Docs</a> •
    <a href="https://moabb.neurotechx.com/docs/install/install.html">Install</a> •
    <a href="https://moabb.neurotechx.com/docs/auto_examples/index.html">Examples</a> •
    <a href="https://moabb.neurotechx.com/docs/paper_results.html">Benchmark</a> •
    <a href="https://moabb.neurotechx.com/docs/dataset_summary.html">Datasets</a>
  </p>
  <p>
    <a href="https://doi.org/10.5281/zenodo.10034223"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10034223.svg" alt="DOI"></a>
    <a href="https://github.com/NeuroTechX/moabb/actions?query=branch%3Adevelop"><img src="https://github.com/NeuroTechX/moabb/workflows/Test/badge.svg" alt="Build Status"></a>
    <a href="https://pypi.org/project/moabb/"><img src="https://img.shields.io/pypi/v/moabb?color=blue&style=flat-square" alt="PyPI"></a>
    <a href="https://pypi.org/project/moabb/"><img src="https://img.shields.io/pypi/v/moabb?label=version&color=orange&style=flat-square" alt="Version"></a>
    <a href="https://pypi.org/project/moabb/"><img src="https://img.shields.io/pypi/pyversions/moabb?style=flat-square" alt="Python versions"></a>
    <a href="https://pepy.tech/project/moabb"><img src="https://pepy.tech/badge/moabb" alt="Downloads"></a>
    <a href="https://github.com/NeuroTechX/moabb/actions/workflows/link-check.yml"><img src="https://github.com/NeuroTechX/moabb/actions/workflows/link-check.yml/badge.svg" alt="Link Check"></a>
  </p>
</div>

## Quickstart


```shell
pip install moabb
```

```python
import moabb
from moabb.datasets import BNCI2014_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.pipelines.features import LogVariance

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

moabb.set_log_level("info")

pipelines = {"LogVar+LDA": make_pipeline(LogVariance(), LDA())}

dataset = BNCI2014_001()
dataset.subject_list = dataset.subject_list[:2]

paradigm = LeftRightImagery(fmin=8, fmax=35)
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset])
results = evaluation.process(pipelines)

print(results.head())
```

For full installation options and troubleshooting, see the [documentation](https://moabb.neurotechx.com/docs/install/install.html).


## Disclaimer

**This is an open science project that may evolve depending on the need of the community.**

## The problem

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

## The solution

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

## Core Team

This project is under the umbrella of [NeuroTechX][link_neurotechx], the international
community for NeuroTech enthusiasts.

The Mother of all BCI Benchmarks was founded by [Alexander Barachant](http://alexandre.barachant.org/) and [Vinay Jayaram][link_vinay].

It is currently maintained by:

* [Sylvain Chevallier](https://sylvchev.github.io/)
* [Bruno Aristimunha](https://bruaristimunha.github.io/)
* [Pierre Guetschel](https://github.com/PierreGtch)
* [Grégoire Cattan](https://github.com/gcattan)
* [Anton Andreev](https://github.com/toncho11)

## Contributors

The MOABB is a community project, and we are always thankful to all the contributors!

<div align="center" class="moabb-contributors">
  <a href="https://github.com/NeuroTechX/moabb/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=NeuroTechX/moabb" alt="MOABB contributors" width="1100" />
  </a>
</div>

## Acknowledgements

MOABB has benefited from the support of the following organizations:

<a href="https://www.dataia.eu/en"><img src="https://www.dataia.eu/themes/dataia/css/images/DATAIA-h-sansfond.png" alt="DATAIA" style="height:60px; background-color:#2e4a7d; padding:10px; border-radius:5px;"/></a>

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

## Cite MOABB

If you use MOABB in your experiments, please cite MOABB and the related publications:

📚 [Full citation guide](https://moabb.neurotechx.com/docs/cite.html)

### Software Citation

#### APA Format

```text
Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J.,
Narayanan, D., Bjareholt, E., Barthelemy, Q., Schirrmeister, R. T., Kobler, R.,
Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V.,
Andreev, A., Thielen, J., Moreau, T., Roy, Y., Jayaram, V., Barachant, A., &
Chevallier, S. (2025). Mother of all BCI Benchmarks (MOABB) (Version 1.4.3).
Zenodo. https://doi.org/10.5281/zenodo.10034223
```

#### BibTeX Format

```bibtex
@software{Aristimunha_Mother_of_all,
  author       = {Aristimunha, Bruno and
                  Carrara, Igor and
                  Guetschel, Pierre and
                  Sedlar, Sara and
                  Rodrigues, Pedro and
                  Sosulski, Jan and
                  Narayanan, Divyesh and
                  Bjareholt, Erik and
                  Barthelemy, Quentin and
                  Schirrmeister, Robin Tibor and
                  Kobler, Reinmar and
                  Kalunga, Emmanuel and
                  Darmet, Ludovic and
                  Gregoire, Cattan and
                  Abdul Hussain, Ali and
                  Gatti, Ramiro and
                  Goncharenko, Vladislav and
                  Andreev, Anton and
                  Thielen, Jordy and
                  Moreau, Thomas and
                  Roy, Yannick and
                  Jayaram, Vinay and
                  Barachant, Alexandre and
                  Chevallier, Sylvain},
  title        = {Mother of all BCI Benchmarks},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.4.3},
  url          = {https://github.com/NeuroTechX/moabb},
  doi          = {10.5281/zenodo.10034223},
}
```

### Scientific Publications

If you want to cite the scientific contributions of MOABB, please use the following papers:

#### MOABB Benchmark Paper

> Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar,
> Bruna Junqueira Lopes, Sébastien Velut, Salim Khazem, Thomas Moreau
>
> **["The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark"](https://cnrs.hal.science/hal-04537061/)**
>
> HAL: hal-04537061

#### Original MOABB Paper

> Vinay Jayaram and Alexandre Barachant
>
> **["MOABB: trustworthy algorithm benchmarking for BCIs"](https://doi.org/10.1088/1741-2552/aadea0)**
>
> Journal of Neural Engineering 15.6 (2018): 066011
>
> [DOI: 10.1088/1741-2552/aadea0](https://doi.org/10.1088/1741-2552/aadea0)

---

📣 **If you publish a paper using MOABB, please [open an issue](https://github.com/NeuroTechX/moabb/issues) to let us know!**
We would love to hear about your work and help you promote it.

## Contact us

If you want to report a problem or suggest an enhancement, we'd love for you to
[open an issue](https://github.com/NeuroTechX/moabb/issues) at this GitHub repository
because then we can get right on it.


[link_alex_b]: http://alexandre.barachant.org/
[link_vinay]: https://www.linkedin.com/in/vinay-jayaram-8635aa25
[link_neurotechx]: http://neurotechx.com/
[link_sylvain]: https://sylvchev.github.io/
[link_bruno]: https://www.linkedin.com/in/bruaristimunha/
[link_igor]: https://www.linkedin.com/in/carraraig/
[link_pierre]: https://www.linkedin.com/in/pierreguetschel/
[link_neurotechx_signup]: https://neurotechx.com/
[link_gitter]: https://app.gitter.im/#/room/#moabb_dev_community:gitter.im
[link_moabb_docs]: https://moabb.neurotechx.com/
[link_arxiv]: https://arxiv.org/abs/1805.06427
[link_jne]: https://doi.org/10.1088/1741-2552/aadea0
