test:
	@python -m pytest

mypy:
	@mypy --ignore-missing-imports miv tests

all:test mypy
ci: test mypy
