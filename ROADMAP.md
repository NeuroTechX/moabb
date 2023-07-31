# Roadmap

## Mother of all BCI Benchmark

Build a comprehensive benchmark of popular BCI algorithms applied on an extensive list of
freely available EEG datasets.

Most of the issues and ideas are discussed on
[Gitter](https://gitter.im/moabb_dev/community) and during office hours (visio meeting
every Thursday at 18:30 GMT+1, you could ask for a link on Gitter channel). The discussion
are reported in the
[dedicated wiki page](https://github.com/NeuroTechX/moabb/wiki/Weekly-MOABB-meeting)

## Short term - what we're working on now

- Backend features for dev: pre-commit using black, isort and prettier, with an updating
  CONTRIBUTING section.
- Including support for more datasets
- Add more classification pipelines

## Medium term - what we're working on next

- Up to date and automatically built documentation
- Having a leaderboard that display the score of all classification pipelines on all
  datasets
- Transfer learning support, we already have learning curve support.
- BIDS compliant formatting

## Long term - what we're working on in the future

- Pytorch support or other backend, different from vanilla scikit-learn.
- Connect the leaderboard with [Papers with code](https://paperswithcode.com/task/eeg)
- Organize code sprint and ML competition
- Support different paradigm (affective BCI)
- Support more than fNIRS and EEG signal
- Organize a resilient and decentralized data sharing, using [dvc](https://dvc.org/) or
  [Datalad](https://www.datalad.org/
