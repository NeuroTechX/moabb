help:
	@echo "venv      - Create spot_venv and install all necessary dependencies."
	@echo "test      - Run tests. Requires virtual env set up."

venv:
	python3 -m venv venv ;\
	. venv/bin/activate ;\
	pip3 install --upgrade pip ;\
	pip3 install cython ;\
	pip3 install -r requirements.txt ;\
	pip3 install -e .

test:
	flake8 moabb ;\
	. venv/bin/activate ;\
	python -m unittest moabb.tests
