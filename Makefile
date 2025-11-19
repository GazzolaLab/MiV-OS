#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* Installation
.PHONY: install
install:
	uv sync

.PHONY: install-dev
install-dev:
	uv sync --dev

.PHONY: install-all
install-all:
	uv sync --all-extras --no-extra mpi --dev

.PHONY: pre-commit-install
pre-commit-install:
	uv run pre-commit install

#* Formatters
.PHONY: formatting
formatting:
	uv run ruff format --config pyproject.toml miv

#* Linting
.PHONY: test
test:
	uv run pytest -c pyproject.toml --cov=miv/core --cov-report=xml tests

.PHONY: test-mpi
test-mpi:
	mpirun -n 4 uv run pytest --with-mpi -c pyproject.toml tests-mpi

.PHONY: check-codestyle
check-codestyle:
	uv run ruff check --config pyproject.toml miv/core

.PHONY: mypy
mypy:
	uv run mypy --config-file pyproject.toml miv/core

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: ruffcache-remove
ruffcache-remove:
	find . | grep -E ".ruff_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove ruffcache-remove
