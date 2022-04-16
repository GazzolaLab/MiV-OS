test:
	@python -m pytest

mypy:
	@mypy --ignore-missing-imports miv

coverage:
	@pytest --cov=miv tests/

all:test mypy
ci: test mypy
