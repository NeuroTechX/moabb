# Contributing

Contributions are always welcome, no matter how small.

The following is a small set of guidelines for how to contribute to the project

## Where to start

### Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By
participating you are expected to adhere to these expectations. Please report unacceptable
behavior to [hi@pushtheworld.us](mailto:hi@pushtheworld.us)

### Contributing on Github

If you're new to Git and want to learn how to fork this repo, make your own additions, and
include those additions in the master version of this project, check out this
[great tutorial](http://blog.davidecoppola.com/2016/11/howto-contribute-to-open-source-project-on-github/).

### Community

This project is maintained by the [NeuroTechX](www.neurotechx.com) community. Join the
[Gitter](https://gitter.im/moabb_dev/community), where discussions about MOABB takes
place.

## How can I contribute?

If there's a feature you'd be interested in building or you find a bug or have a
suggestion on how to improve the project, go ahead! Let us know on the
[Gitter](https://gitter.im/moabb_dev/community) or [open an issue](../../issues) so others
can follow along and we'll support you as much as we can. When you're finished submit a
pull request to the master branch referencing the specific issue you addressed.

### Steps to Contribute

1. Look for open issues or open one
1. Discuss the problem and or propose a solution
1. Fork it! (and clone fork locally)
1. Branch from `develop`: `git checkout --track develop`
1. [Setup development environment](#setup-development-environment)
1. Create your feature branch: `git checkout -b my-new-feature`
1. Make changes
1. Commit your changes: `git commit -m 'Add some feature'`
1. Don't forget to fix issues from `pre-commit` pipeline (either add changes made by hooks
   or fix them manually in case of `flake8`)
1. Push to the branch: `git push origin my-new-feature`
1. Submit a pull request. Make sure it is based on the `develop` branch when submitting!
   :D
1. Don't forget to update the
   [what's new](http://moabb.neurotechx.com/docs/whats_new.html) and
   [documentation](http://moabb.neurotechx.com/docs/index.html) pages if needed

## Setup development environment

1. install `poetry` (only once per machine):\
   `curl -sSL https://install.python-poetry.org | python3 -`\
   or [checkout installation instruction](https://python-poetry.org/docs/#installation) or
   use [conda forge version](https://anaconda.org/conda-forge/poetry)
1. (Optional, skip if not sure) Disable automatic environment creation:\
   `poetry config virtualenvs.create false`
1. install all dependencies in one command (have to be run in thibe project directory):\
   `poetry install`
1. install `pre-commit` hooks to git repo:\
   `pre-commit install`
1. you are ready to code!

_Note 1:_\
Your first commit will trigger `pre-commit` to download [Code Quality tools](#tools-used).
That's OK and it is intended behavior. This will be done once per machine automatically.

_Note 2:_\
By default `poetry` creates separate Python virtual environment for every project ([more details in documentation](https://python-poetry.org/docs/managing-environments/)).
If you use `conda` or any other way to manage different environments by hand - you need to
disable `poetry` environment creation. Also in this case be careful with version of Python
in your environment - it has to satisfy requirements stated in `pyproject.toml`. In case you
disable `poetry` you are in charge of this.

### Tools used

MOABB uses [poetry](https://python-poetry.org/) for dependency management. This tool
enables one to have a reproducible environment on all popular OS (Linux, MacOS and
Windows) and provides easy publishing pipeline.

Another tool that makes development more stable is [pre-commit](https://pre-commit.com/).
It automatically runs variety of Code Quality instruments against the code you produced.

For Code Quality verification, we use:

- [black](https://github.com/psf/black) - Python code formatting
- [isort](https://github.com/timothycrosley/isort) - imports sorting and grouping
- [flake8](https://gitlab.com/pycqa/flake8) - code style checking
- [prettier](https://github.com/prettier/prettier) - `.yml` and `.md` files formatting

### Generate the documentation

To generate a local version of the documentation:

```
cd docs
make html
```
