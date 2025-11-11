<p align="center" style="font-size: 40px;">
  <span style="font-size: 70px; font-weight: 1000;">Mother of all BCI Benchmarks</span>   <img src="https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/_static/moabb_logo.svg" width="400" height="400" style="display: block; margin: auto;" />
  Build a comprehensive benchmark of popular Brain-Computer Interface (BCI) algorithms applied on an extensive list of freely available EEG datasets.
</p>


## Disclaimer

**This is an open science project that may evolve depending on the need of the
community.**



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10034224.svg)](https://doi.org/10.5281/zenodo.10034224)
[![Build Status](https://github.com/NeuroTechX/moabb/workflows/Test/badge.svg)](https://github.com/NeuroTechX/moabb/actions?query=branch%3Amaster)
[![PyPI](https://img.shields.io/pypi/v/moabb?color=blue&style=plastic)](https://img.shields.io/pypi/v/moabb)
[![Downloads](https://pepy.tech/badge/moabb)](https://pepy.tech/project/moabb)


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

## Core Team

This project is under the umbrella of [NeuroTechX][link_neurotechx], the international
community for NeuroTech enthusiasts.

The project is currently maintained by:

<table style="text-align: center;">
  <thead>
    <tr>
      <th>Sylvain Chevallier</th>
      <th>Bruno Aristimunha</th>
      <th>Igor Carrara</th>
      <th>Pierre Guetschel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 0 7px;"><img src="https://avatars.githubusercontent.com/u/5824988?s=150&amp;v=4" alt="Sylvain Chevallier"></td>
      <td style="padding: 0 7px;"><img src="https://avatars.githubusercontent.com/u/42702466?s=150&amp;v=4" alt="Bruno Aristimunha"></td>
      <td style="padding: 0 7px;"><img src="https://avatars.githubusercontent.com/u/94047258?s=150&amp;v=4" alt="Igor Carrara"></td>
      <td style="padding: 0 7px;"><img src="https://avatars.githubusercontent.com/u/25532709?s=150&amp;v=4" alt="Pierre Guetschel"></td>
</tr>
  </tbody>
</table>

The Mother of all BCI Benchmarks was founded by Alexander Barachant and Vinay Jayaram, who
are experts in the field of Brain-Computer Interfaces (BCI). At the moment, both work as
Research Scientists at Meta.

<table style="text-align: center;">
  <thead>
    <tr>
      <th>Alexander Barachant</th>
      <th>Vinay Jayaram</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td style="padding: 0 15px;"><img src="http://alexandre.barachant.org/images/avatar.jpg" alt="Alexander Barachant" width="150" height="150"></td>
<td style="padding: 0 15px;"><img src="https://beetl.ai/static/media/vinay.217f36bc.jpeg" alt="Vinay Jayaram" width="150" height="150"></td></tr>
  </tbody>
</table>

## Contributors

The MOABB is a community project, and we are always thankful to all the contributors!

<div id="contributors-container"></div>

<script>
const endpoint = 'https://api.github.com/repos/NeuroTechX/moabb/contributors';
const container = document.getElementById('contributors-container');
const filterList = ["bruAristimunha", "sylvchev", "carraraig", "pierreGtch", "sara04", "pre-commit-ci[bot]", "dependabot[bot]", "alexandrebarachant", "vinay-jayaram"];
fetch(endpoint)
  .then(response => response.json())
  .then(contributors => {
    const filteredContributors = contributors.filter(contributor => !filterList.includes(contributor.login));    filteredContributors.forEach(contributor => {
      const link = document.createElement('a');
      link.href = contributor.html_url;
      link.target = '_blank';
      const img = document.createElement('img');
      img.src = contributor.avatar_url;
      img.alt = contributor.login;
      img.style.width = '100px';
      img.style.height = '100px';
      img.style.objectFit = 'cover';
      img.style.borderRadius = '50%';
      link.appendChild(img);
      container.appendChild(link);
    });
  });
</script>

Special acknowledge for the extra MOABB contributors:

<table style="text-align: center;">
  <thead>
    <tr>
      <th>Pedro Rodrigues</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td style="padding: 0 15px;"><img src="https://avatars.githubusercontent.com/u/4588557?v=4" alt=" Pedro L. C. Rodrigues" width="150" height="150"></td>
  </tbody>
</table>

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

## Citing MOABB and related publications

If you use MOABB in your experiments, please cite this library when
publishing a paper to increase the visibility of open science initiatives:

* Here is the APA version:
```
Aristimunha, B., Carrara, I., Guetschel, P., Sedlar, S., Rodrigues, P., Sosulski, J., Narayanan, D., Bjareholt, E., Barthelemy, Q., Schirrmeister, R. T., Kobler, R., Kalunga, E., Darmet, L., Gregoire, C., Abdul Hussain, A., Gatti, R., Goncharenko, V., Thielen, J., Moreau, T., Roy, Y., Jayaram, V., Barachant, A., & Chevallier, S. (2025).
Mother of all BCI Benchmarks (MOABB), 2025. DOI: 10.5281/zenodo.10034223.
```

and the Bibtex version:

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
                      Thielen, Jordy and
                      Moreau, Thomas and
                      Roy, Yannick and
                      Jayaram, Vinay and
                      Barachant, Alexandre and
                      Chevallier, Sylvain},
            title        = {Mother of all BCI Benchmarks},
            year         = 2025,
            publisher    = {Zenodo},
            version      = {v1.2.0},
            url = {https://github.com/NeuroTechX/moabb},
            doi = {10.5281/zenodo.10034223},
    }
```

If you want to cite the scientific contributions of MOABB, you could use the following paper:

> Sylvain Chevallier, Igor Carrara, Bruno Aristimunha, Pierre Guetschel, Sara Sedlar, Bruna Junqueira Lopes, Sébastien Velut, Salim Khazem, Thomas Moreau
> ["The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark"](https://cnrs.hal.science/hal-04537061/)
> HAL: hal-04537061.

> Vinay Jayaram and Alexandre Barachant.
> ["MOABB: trustworthy algorithm benchmarking for BCIs."](http://iopscience.iop.org/article/10.1088/1741-2552/aadea0/meta)
> Journal of neural engineering 15.6 (2018): 066011.
> [DOI](https://doi.org/10.1088/1741-2552/aadea0)

If you publish a paper using MOABB, please contact us on [gitter][link_gitter] or open an
issue! We would love to hear about your work and help you promote it.


## Contact us

If you want to report a problem or suggest an enhancement, we'd love for you to
[open an issue](https://github.com/NeuroTechX/moabb/issues) at this GitHub repository
because then we can get right on it.

For a less formal discussion or exchanging ideas, you can also reach us on the Github or join our weekly office hours! This an open video meeting
happening on a [regular basis](https://github.com/NeuroTechX/moabb/issues/191), please ask
the link on the gitter channel. We are also on NeuroTechX Slack channel
[#moabb][link_neurotechx_signup].


[link_alex_b]: http://alexandre.barachant.org/
[link_vinay]: https://www.linkedin.com/in/vinay-jayaram-8635aa25
[link_neurotechx]: http://neurotechx.com/
[link_sylvain]: https://sylvchev.github.io/
[link_bruno]: https://www.linkedin.com/in/bruaristimunha/
[link_igor]: https://www.linkedin.com/in/carraraig/
[link_pierre]: https://www.linkedin.com/in/pierreguetschel/
[link_neurotechx_signup]: https://neurotechx.com/
[link_gitter]: https://app.gitter.im/#/room/#moabb_dev_community:gitter.im
[link_moabb_docs]: https://neurotechx.github.io/moabb/
[link_arxiv]: https://arxiv.org/abs/1805.06427
[link_jne]: http://iopscience.iop.org/article/10.1088/1741-2552/aadea0/meta
