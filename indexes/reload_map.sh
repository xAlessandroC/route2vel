#!/bin/bash

# Ensure the script is executed with one argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <map_name>"
    exit 1
fi

# Assign input to mapNameOsrm variable
input_string=$1
IFS='_' read -r -a tokens <<< "$input_string"
mapNameOsrm="${tokens[-1]}.osrm"

# Container name
CONTAINER_NAME="osrm_backend-routed"

# Get the directory of the script
scriptDirectory="/home/routevel/route_vel_service/route2vel/indexes/$1"

# Stop the container if it is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping container $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME
fi

# Remove the container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Removing container $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME
fi

# Create and start the container with the specified parameters
echo "Creating and starting container $CONTAINER_NAME with map $mapNameOsrm..."
docker create --name $CONTAINER_NAME -p 5000:5000 -v "$scriptDirectory:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm"

echo "docker create --name $CONTAINER_NAME -p 5000:5000 -v "$scriptDirectory:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm""

# Start the newly created container
docker start $CONTAINER_NAME

echo "Container $CONTAINER_NAME has been restarted with map $mapNameOsrm."
``
