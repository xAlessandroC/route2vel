$MAP_NAME = "nord-est-latest.osm.pbf"

if (-not(Test-Path -Path "${PWD}/osrm_files" -PathType Container)) {
    New-Item "${PWD}/osrm_files" -ItemType Directory -ea 0
}
if (-not(Test-Path -Path "${PWD}/resources/osm" -PathType Container)) {
    New-Item "${PWD}/resources/osm" -ItemType Directory -ea 0
}
$MAP_PATH = "${PWD}/resources/osm/$MAP_NAME"
if (-not(Test-Path -Path $MAP_PATH -PathType Leaf)) {
    Write-Host "Downloading $MAP_NAME..."
    Invoke-WebRequest -OutFile $MAP_PATH -Uri "http://download.geofabrik.de/europe/italy/$MAP_NAME"
    Write-Host "Downloaded $MAP_NAME"
} else {
    $lastModified = (Get-Item $MAP_PATH).LastWriteTime
    $currentDate = Get-Date
    $daysDifference = ($currentDate - $lastModified).Days

    if ($daysDifference -ge 14) {
        $choice = Read-Host "$MAP_NAME is $daysDifference days old. Do you want to redownload it? (y/N)"
        if ($choice -eq "Y" -or $choice -eq "y") {
            Write-Host "Redownloading $MAP_NAME..."
            Invoke-WebRequest -OutFile $MAP_PATH -Uri "http://download.geofabrik.de/europe/italy/$MAP_NAME"
            Write-Host "Redownloaded $MAP_NAME"
        } else {
            Write-Host "Ok. It is recommended to keep the map files up to date, so download the updated files when possible."
        }
    }
}
if (-not(Test-Path -Path "${PWD}/osrm_files/$MAP_NAME" -PathType Leaf)) {
    New-Item -ItemType HardLink -Path "${PWD}/osrm_files/$MAP_NAME" -Target "${PWD}/resources/osm/$MAP_NAME"
    Write-Host "Created link to $MAP_NAME"
}

$CONTAINER_NAME = "osrm_backend-routed"

if (!(docker ps -a | Select-String $CONTAINER_NAME)) {
    $mapNameOsrm = $MAP_NAME -Replace ".osm.pbf",".osrm"

    docker run --name osrm_backend-extract -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-extract -p /opt/car.lua "/data/$MAP_NAME"
    docker run --name osrm_backend-partition -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-partition "/data/$mapNameOsrm"
    docker run --name osrm_backend-customize -t -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-customize "/data/$mapNameOsrm"
    Write-Host "Creating container $CONTAINER_NAME"
    docker create --name $CONTAINER_NAME -p 5000:5000 -v "${PWD}/osrm_files:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm"
}
Write-Host "Starting container $CONTAINER_NAME"
docker start $CONTAINER_NAME
