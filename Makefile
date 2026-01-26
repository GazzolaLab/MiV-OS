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
	uv sync --all-extras --no-extra mpi --dev --all-groups

.PHONY: pre-commit-install
pre-commit-install:
	uv run --frozen pre-commit install

#* Formatters
.PHONY: formatting
formatting:
	uv run --frozen ruff format --config pyproject.toml miv tests

#* Linting
.PHONY: test
test:
	uv run --frozen pytest -c pyproject.toml --cov=miv/core --cov-report=xml tests

.PHONY: test-all
test-all:
	# Clean up any existing coverage files
	rm -f .coverage .coverage.*
	# Run MPI tests with parallel coverage
	# Disable for now. mpi test on github-action is causing too much issue
	# mpirun --allow-run-as-root -n 2 uv run --frozen coverage run -p -m pytest --with-mpi -c pyproject.toml tests-mpi
	# Run non-MPI tests with parallel coverage
	uv run --frozen coverage run -p -m pytest -c pyproject.toml tests
	# Combine all coverage files
	uv run --frozen coverage combine
	# Generate XML and text reports
	uv run --frozen coverage xml
	uv run --frozen coverage report

.PHONY: view-coverage
view-coverage:
	uv run --frozen coverage html
	open htmlcov/index.html

.PHONY: check-codestyle
check-codestyle:
	uv run --frozen ruff check --config pyproject.toml miv/core

.PHONY: mypy
mypy:
	uv run --frozen mypy --config-file pyproject.toml miv/core

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
