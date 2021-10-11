.PHONY: test


clean: clean-pyc clean-test ## remove all test, coverage and Python artifacts

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test: ## run tests quickly with the default Python
	python -m unittest discover -s tests

install:
	pip install -e .
