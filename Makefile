# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install:
	pip install poetry tox
	poetry shell
	poetry install

check_code:
	@pre-commit run --all-files


black:
	@black scripts/* pelage/*.py testsma/*.py

test:
	@coverage run --data-file=".coverage/pytest" -m pytest tests/*.py
	@coverage report --data-file=".coverage/pytest" -m --omit="${VIRTUAL_ENV}/lib/python*"
	@coverage erase --data-file=".coverage/pytest"

clean:
	@rm -f */version.txt
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr pelage-*.dist-info
	@rm -fr pelage.egg-info

tox:
	@tox run-parallel
	@coverage report --data-file=".coverage/.coverage" --show-missing  --precision=3


all: clean test tox publish_checks check_code pre_commit

pre_commit:
	pre-commit run --files docs/*

render docs:
	quartodoc build --config docs/_quarto.yml
	quarto render docs
	quarto render docs/notebooks/initial_readme.ipynb --to gfm --output README.md
	mv docs/README.md README.md


publish_checks:
	make render docs
	@python -m doctest pelage/checks.py
	@echo "doctest check passed"
