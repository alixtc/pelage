# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

install:
	uv venv --python 3.11
	. .venv/bin/activate
	uv sync --all-groups --python 3.11

test:
	@coverage run --data-file=".coverage/pytest" -m pytest
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

all: clean test tox render-docs doctest pre_commit

pre_commit:
	git ls-files -- 'docs/*' | xargs pre-commit run --files

render-docs:
	quartodoc build --config docs/_quarto.yml
	quarto render docs
	quarto render docs/index.ipynb --to gfm --output README.md
	mv docs/README.md README.md
	git ls-files -- 'docs/*' | xargs pre-commit run --files

doctest:
	@python -m doctest pelage/checks/*.py
	@echo "doctest check passed"
