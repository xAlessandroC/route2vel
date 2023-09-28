$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvDir = "$scriptDirectory/.venv"

# Create a virtual environment if it doesn't exist
if (-Not (Test-Path -Path $venvDir -PathType Container)) {
    Write-Host "Creating venv..."
    python -m venv $venvDir
}

Write-Host "Activating venv..."
& "$venvDir\Scripts\Activate.ps1"

Write-Host "Creating and running containers..."
& "$scriptDirectory\run_osrm.ps1"

# Install dependencies using pip if requirements.txt exists
Write-Host "Checking requirements"
pip install -r "$scriptDirectory\requirements.txt" | Select-String -NotMatch -Pattern "Requirement already satisfied" | Write-Host

python $scriptDirectory/src/web-ui.py
