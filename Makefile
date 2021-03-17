help:
	@echo "venv      - Create virtual environment and install all necessary dependencies."
	@echo "venv_ext  - Create virtual environment and install all necessary dependencies"
	@echo "            and the dependencies needed for examples with external code."
	@echo "test      - Run tests. Requires virtual env set up."

venv:
	python3 -m venv venv ;\
	. venv/bin/activate ;\
	pip3 install --upgrade pip ;\
	pip3 install cython ;\
	pip3 install -r requirements.txt ;\
	pip3 install -e .

venv_ext:
	python3 -m venv venv ;\
	. venv/bin/activate ;\
	pip3 install --upgrade pip ;\
	pip3 install cython ;\
	pip3 install -r requirements.txt ;\
	pip3 install -r examples/external/requirements_external.txt ;\
	pip3 install -e .

test:
	flake8 moabb ;\
	. venv/bin/activate ;\
	python -m unittest moabb.tests
