@echo off

REM Ensure the script is executed with one argument
if "%~1"=="" (
    echo Usage: %0 ^<map_name^>
    exit /b 1
)

REM Assign input to mapNameOsrm variable
set input_string=%~1
for /f "tokens=1,* delims=_" %%a in ("%input_string%") do (
    set last_token=%%b
)
set mapNameOsrm=%last_token%.osrm

REM Container name
set CONTAINER_NAME=osrm_backend-routed

REM Get the directory of the script
set scriptDirectory=E:\indexes\%input_string%

REM Stop the container if it is running
FOR /F "tokens=*" %%i IN ('docker ps -q -f "name=%CONTAINER_NAME%"') DO (
    echo Stopping container %CONTAINER_NAME%...
    docker stop %CONTAINER_NAME%
)

REM Remove the container if it exists
FOR /F "tokens=*" %%i IN ('docker ps -aq -f "name=%CONTAINER_NAME%"') DO (
    echo Removing container %CONTAINER_NAME%...
    docker rm %CONTAINER_NAME%
)

REM Create and start the container with the specified parameters
echo Creating and starting container %CONTAINER_NAME% with map %mapNameOsrm%...
docker create --name %CONTAINER_NAME% -p 5000:5000 -v "%scriptDirectory%:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/%mapNameOsrm%"

echo docker create --name %CONTAINER_NAME% -p 5000:5000 -v "%scriptDirectory%:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/%mapNameOsrm%"

REM Start the newly created container
docker start %CONTAINER_NAME%

echo Container %CONTAINER_NAME% has been restarted with map %mapNameOsrm%.
