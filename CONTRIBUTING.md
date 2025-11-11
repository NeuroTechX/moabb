# Contributing

🎉🥂 First off, thanks for taking the time to contribute! 🎉🥂

Contributions are always welcome, no matter how small!

If you think you can help in any of the areas of MOABB (and we bet you can) or in any of
the many areas that we haven't yet thought of (and here we're _sure_ you can) then please
check out our [roadmap](https://github.com/NeuroTechX/moabb/blob/master/ROADMAP.md).

Please note that it's very important to us that we maintain a positive and supportive
environment for everyone who wants to participate. When you join us we ask that you follow
our [code of conduct](https://github.com/NeuroTechX/moabb/blob/master/CODE_OF_CONDUCT.md)
in all interactions both on and offline.

The following is a small set of guidelines for how to contribute to the project

## Where to start

### Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](https://github.com/NeuroTechX/moabb/blob/master/CODE_OF_CONDUCT.md). By
participating you are expected to adhere to these expectations. Please report unacceptable
behavior.

### Contributing on Github

If you're new to Git and want to learn how to fork this repo, make your own additions, and
include those additions in the master version of this project, check out this
[great tutorial](https://hackerbits.com/how-to-contribute-to-an-open-source-project-on-github-davide-coppola/).

### Community

This project is maintained by the [NeuroTechX](http://www.neurotechx.com) community.

## How can I contribute?

If there's a feature you'd be interested in building or you find a bug or have a
suggestion on how to improve the project, go ahead! Let us know on the
[open an issue](https://github.com/NeuroTechX/moabb/issues) so others
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

(setup-development-environment)=
## Setup development environment

1. Install pre-commit to start to code:\
   `pip install pre-commit`
2. Install the pre-commit hooks:\
   `pre-commit install`
3. you are ready to code!

_Note 1:_\
Your first commit will trigger `pre-commit` to download [Code Quality tools](#tools-used).
That's OK and it is intended behavior. This will be done once per machine automatically.

_Note 2 (deep learning):_\
In case you want to install the optional deep learning dependencies (i.e. `pip install .[deeplearning]`),

(tools-used)=
### Tools used

MOABB uses [`pre-commit`](https://pre-commit.com/). It automatically runs variety of Code Quality
instruments against the code you produced.

For Code Quality verification, we use:

- [`black`](https://github.com/psf/black) - Python code formatting
- [`isort`](https://github.com/timothycrosley/isort) - imports sorting and grouping
- [`flake8`](https://gitlab.com/pycqa/flake8) - code style checking
- [`prettier`](https://github.com/prettier/prettier) - `.yml` and `.md` files formatting
- and more checkers.


### Generate the documentation

To generate a local version of the documentation:

```bash
cd docs
make html
```
