# Makefile

.PHONY: all install test clean venv

PYTHON_EXEC = ./.venv/Scripts/python
PIP_EXEC = ./.venv/Scripts/pip

all: install test

venv:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Virtual environment created at ./.venv"

install: venv
	@echo "Installing dependencies from requirements.txt..."
	$(PIP_EXEC) install -r requirements.txt
	@echo "Dependencies installed."

test:
	@echo "Running NTRU tests..."
	@$(PYTHON_EXEC) -m unittest test_ntru.py

clean:
	@echo "Cleaning up virtual environment and build artifacts..."
	rm -rf .venv
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean up complete."

# Help message (optional, but good practice)
help:
	@echo "NTRU Cryptography Project Makefile Commands:"
	@echo "  all     : Installs dependencies and runs tests (default)."
	@echo "  venv    : Creates a local Python virtual environment."
	@echo "  install : Installs required Python packages into the virtual environment."
	@echo "  test    : Runs the automated tests for the NTRU implementation."
	@echo "  clean   : Removes the virtual environment and Python cache files."
	@echo "  help    : Displays this help message."