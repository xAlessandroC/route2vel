#!/bin/bash


echo "going to IT"
MAP_NAME="it.osm.pbf"


CONTAINER_NAME="osrm_backend-routed"

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

mapNameOsrm="${MAP_NAME/.osm.pbf/.osrm}"

docker create --name "$CONTAINER_NAME" -p 5000:5000 -v "/home/routevel/route_vel_service/route2vel/indexes/eu_it:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm"

echo "docker create --name "$CONTAINER_NAME" -p 5000:5000 -v "/home/routevel/route_vel_service/route2vel/indexes/eu_it:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm""

echo "C: Starting existing container $CONTAINER_NAME"
docker start "$CONTAINER_NAME"
