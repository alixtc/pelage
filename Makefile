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
	@coverage run --data-file=".coverage/.coverage" -m pytest tests/*.py
	@coverage report --data-file=".coverage/.coverage" -m --omit="${VIRTUAL_ENV}/lib/python*"

clean:
	@rm -f */version.txt
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr pelage-*.dist-info
	@rm -fr pelage.egg-info

tox:
	@tox run


all: clean test tox publish_checks check_code


render docs:
	quartodoc build --config docs/_quarto.yml
	quarto render docs
	quarto render docs/notebooks/initial_readme.ipynb --to gfm --output README.md
	pre-commit run --files docs/*


publish_checks:
	make render docs
	@python -m doctest pelage/checks.py
	@echo "doctest check passed"
