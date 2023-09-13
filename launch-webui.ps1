$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvDir = '.venv'

# Change to the script's directory
Set-Location $scriptDirectory

# Create a virtual environment if it doesn't exist
if (-Not (Test-Path -Path $venvDir -PathType Container)) {
    python -m venv $venvDir
}

& "$venvDir\Scripts\Activate.ps1"

& ".\run_osrm.ps1"

# Install dependencies using pip if requirements.txt exists
pip install -r "requirements.txt" | Select-String -NotMatch -Pattern "Requirement already satisfied"

Set-Location src
python web-ui.py
Set-Location ..
