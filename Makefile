#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = inr-apodizations
ENV_NAME = inr-apodizations
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = conda run -n $(ENV_NAME) python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) inr_apodizations/dataset.py


## Run create_delayed_samples_dataset script (generates delayed samples dataset)
.PHONY: run-delayed-samples
run-delayed-samples:
	$(PYTHON_INTERPRETER) scripts/create_delayed_samples_dataset.py

## Run numeric phantom evaluation script
.PHONY: run-phantom-eval
run-phantom-eval:
	$(PYTHON_INTERPRETER) sandbox\inr_das_experiment\evaluation\numeric_phantom\evaluate_apodizations.py

## Run FFT profile evaluation script
.PHONY: run-fft-profile-eval
run-fft-profile-eval:
	$(PYTHON_INTERPRETER) sandbox\inr_das_experiment\evaluation\evaluate_apodization_profile_fft.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
