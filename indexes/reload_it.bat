@echo off

echo going to IT
set "MAP_NAME=it.osm.pbf"

set "CONTAINER_NAME=osrm_backend-routed"

REM Stop the container if it is running
FOR /F "tokens=*" %%i IN ('docker ps -q -f name=%CONTAINER_NAME%') DO (
    echo Stopping container %CONTAINER_NAME%...
    docker stop %CONTAINER_NAME%
)

REM Remove the container if it exists
FOR /F "tokens=*" %%i IN ('docker ps -aq -f name=%CONTAINER_NAME%') DO (
    echo Removing container %CONTAINER_NAME%...
    docker rm %CONTAINER_NAME%
)

set "mapNameOsrm=%MAP_NAME:.osm.pbf=.osrm%"

docker create --name "%CONTAINER_NAME%" -p 5000:5000 -v "E:\indexes\eu_it:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/%mapNameOsrm%"

echo docker create --name "%CONTAINER_NAME%" -p 5000:5000 -v "E:\indexes\eu_it:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/%mapNameOsrm%"

echo C: Starting existing container %CONTAINER_NAME%
docker start "%CONTAINER_NAME%"
