#!/usr/bin/env sh

# Name of the virtual environment
VENV_NAME="ae_autoencoder"

# Create the virtual environment
python3 -m venv "$VENV_NAME"

# Activate the virtual environment
source "$VENV_NAME"/bin/activate

echo "Virtual environment $VENV_NAME activated."
