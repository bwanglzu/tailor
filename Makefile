# Run:
#   make help
#
# for a description of the available targets


# ------------------------------------------------------------------------- Help target

TARGET_MAX_CHAR_NUM=20
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

## Show this help message
help:
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  ${YELLOW}%-$(TARGET_MAX_CHAR_NUM)s${RESET} ${GREEN}%s${RESET}\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)


# ------------------------------------------------------------------------ Clean target

## Delete temp operational stuff like artifacts, test outputs etc
clean:
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -f .coverage
	rm -f .coverage.*
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/


# --------------------------------------------------------- Environment related targets

## Create a virtual environment
env:
	python3.9 -m venv .venv
	source .venv/bin/activate
	pip install -U pip

## Install package
install:
	pip install --no-cache-dir .

## Install test dependencies
install-test:
	pip install --no-cache-dir -r .github/requirements-test.txt

## Install CI/CD dependencies
install-dev:
	pip install --no-cache-dir -r .github/requirements-cpu.txt

## Install CI/CD cuda dependencies
install-dev-cuda:
	pip install --no-cache-dir -r .github/requirements-cuda.txt

## Install pre-commit hooks
pre-commit:
	pip install pre-commit
	pre-commit install

## Bootstrap a dev environment
init: pre-commit install-dev


# ----------------------------------------------------------------------- Build targets

## Build wheel
build:
	python setup.py bdist_wheel

## Build source dist
build-sdist:
	python setup.py sdist


# ---------------------------------------------------------------- Test related targets

PYTEST_ARGS = --show-capture no --full-trace --verbose --cov tailor/ --cov-report term-missing --cov-report html

## Run tests
test:
	pytest $(TESTS_PATH) $(PYTEST_ARGS)

# ---------------------------------------------------------- Code style related targets

SRC_CODE = tailor/ tests/ setup.py

## Run the flake linter
flake:
	flake8 $(SRC_CODE)

## Run the black formatter
black:
	black $(SRC_CODE)

## Dry run the black formatter
black-check:
	black --check $(SRC_CODE)

## Run the isort import formatter
isort:
	isort $(SRC_CODE)

## Dry run the isort import formatter
isort-check:
	isort --check $(SRC_CODE)

## Run the mypy static type checker
mypy:
	mypy $(SRC_CODE)

## Format source code
format: black isort

## Check code style
style: flake black-check isort-check # mypy