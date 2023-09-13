#!/bin/bash

script_directory=$(dirname "$0")
venv_dir=".venv"

# Change to the script's directory
cd "$script_directory"

# Create a virtual environment if it doesn't exist
if [ ! -d "$venv_dir" ]; then
    python -m venv "$venv_dir"
fi

# Activate the virtual environment
source "$venv_dir/bin/activate"

bash run_osrm.sh

# Install dependencies using pip if requirements.txt exists
pip install -r "requirements.txt" | grep -v 'Requirement already satisfied'

cd src
python web-ui.py