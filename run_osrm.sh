#!/bin/bash

MAP_NAME="nord-est-latest.osm.pbf"
MAP_PATH="${PWD}/resources/osm/$MAP_NAME"

if [ ! -d "${PWD}/osrm_files" ]; then
    mkdir "${PWD}/osrm_files"
fi

if [ ! -d "${PWD}/resources/osm" ]; then
    mkdir "${PWD}/resources/osm"
fi

if [ ! -f "$MAP_PATH" ]; then
    echo "Downloading $MAP_NAME..."
    wget -O "$MAP_PATH" "http://download.geofabrik.de/europe/italy/$MAP_NAME"
    echo "Downloaded $MAP_NAME"
else
    # todo: replace date modified check with MD5 check
    last_modified=$(date -r "$MAP_PATH" +%s)
    current_date=$(date +%s)
    days_difference=$(( (current_date - last_modified) / 86400 ))

    if [ "$days_difference" -ge 14 ]; then
        read -p "$MAP_NAME is $days_difference days old. Do you want to redownload it? (y/N): " choice
        if [ "$choice" = "Y" ] || [ "$choice" = "y" ]; then
            echo "Redownloading $MAP_NAME..."
            wget -O "$MAP_PATH" "http://download.geofabrik.de/europe/italy/$MAP_NAME"
            echo "Redownloaded $MAP_NAME"
        else
            echo "Ok. It is recommended to keep the map files up to date, so download the updated files when possible."
        fi
    fi
fi

if [ ! -f "${PWD}/osrm_files/$MAP_NAME" ]; then
    ln "${PWD}/resources/osm/$MAP_NAME" "${PWD}/osrm_files/$MAP_NAME"
    echo "Created link to $MAP_NAME"
fi

CONTAINER_NAME="osrm_backend-routed"

if ! docker ps -a | grep -q "$CONTAINER_NAME"; then
    mapNameOsrm="${MAP_NAME/.osm.pbf/.osrm}"

    docker run --name osrm_backend-extract -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-extract -p /opt/car.lua "/data/$MAP_NAME"
    docker run --name osrm_backend-partition -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-partition "/data/$mapNameOsrm"
    docker run --name osrm_backend-customize -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-customize "/data/$mapNameOsrm"
    echo "Creating container $CONTAINER_NAME"
    docker create --name "$CONTAINER_NAME" -p 5000:5000 -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm"
fi
echo "Starting existing container $CONTAINER_NAME"
docker start "$CONTAINER_NAME"