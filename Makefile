# ----------------------------------
#          INSTALL & TEST
# ----------------------------------

pre_install:
	pyenv virtualenv 3.12 FC3.12
	pyenv virtualenv 3.11 FC3.11
	pyenv virtualenv 3.10 FC3.10
	pyenv virtualenv 3.9 FC3.9
	echo "Run the following commands:
		pyenv local FC3.10
		pyenv shell FC3.10
	"

install:
	pip install poetry tox
	poetry shell
	poetry install

check_code:
	@pre-commit run --all-files

black:
	@black scripts/* furcoat/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr furcoat-*.dist-info
	@rm -fr furcoat.egg-info


all: clean install test check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
