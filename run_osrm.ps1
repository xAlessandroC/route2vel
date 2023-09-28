$MAP_NAME = "nord-est-latest.osm.pbf"
$MAP_URL = "http://download.geofabrik.de/europe/italy/$MAP_NAME"
$MD5_URL = "$MAP_URL.md5"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Definition

$MAP_PATH = "$scriptDirectory/resources/osm/$MAP_NAME"
$MD5_PATH = "$scriptDirectory/resources/md5/$MAP_NAME.md5"

if (-not(Test-Path -Path "$scriptDirectory/osrm_files" -PathType Container)) {
    New-Item "$scriptDirectory/osrm_files" -ItemType Directory -ea 0
}
if (-not(Test-Path -Path "$scriptDirectory/resources/osm" -PathType Container)) {
    New-Item "$scriptDirectory/resources/osm" -ItemType Directory -ea 0
}
if (-not(Test-Path -Path "$scriptDirectory/resources/md5" -PathType Container)) {
    New-Item "$scriptDirectory/resources/md5" -ItemType Directory -ea 0
}

function DownloadFile($url, $targetFile) {
    Start-BitsTransfer -Source $url -Destination $targetFile
}

DownloadFile $MD5_URL $MD5_PATH
$downloadedMd5 = ($(Get-Content $MD5_PATH) -Split "\s+")[0]

if (-not(Test-Path -Path $MAP_PATH -PathType Leaf)) {
    Write-Host "Downloading $MAP_NAME..."
    DownloadFile $MAP_URL $MAP_PATH
    Write-Host "Downloaded $MAP_NAME"
} else {
    $existingHash = $(Get-FileHash $MAP_PATH -Algorithm MD5).Hash.ToLower()

    Write-Host "Existing map file hash: $existingHash"
    Write-Host "Up to date hash is:     $downloadedMd5"

    if (!($downloadedMd5 -ieq $existingHash)) {
        $choice = Read-Host "$MAP_NAME has wrong hash (outdated or wrong download). Do you want to redownload it? (y/N)"
        if ($choice -eq "Y" -or $choice -eq "y") {
            Write-Host "Redownloading $MAP_NAME..."
            Remove-Item $MAP_PATH
            DownloadFile $MAP_URL $MAP_PATH
            Write-Host "Redownloaded $MAP_NAME"
        } else {
            Write-Host "Ok. It is recommended to keep the map files up to date, so download the updated files when possible."
        }
    }
}
if (-not(Test-Path -Path "$scriptDirectory/osrm_files/$MAP_NAME" -PathType Leaf)) {
    New-Item -ItemType HardLink -Path "$scriptDirectory/osrm_files/$MAP_NAME" -Target "$scriptDirectory/resources/osm/$MAP_NAME"
    Write-Host "Created link to $MAP_NAME"
}

$CONTAINER_NAME = "osrm_backend-routed"

if (!(docker ps -a | Select-String $CONTAINER_NAME)) {
    $mapNameOsrm = $MAP_NAME -Replace ".osm.pbf",".osrm"

    docker run --name osrm_backend-extract -t -v "$scriptDirectory/osrm_files:/data" "osrm/osrm-backend" osrm-extract -p /opt/car.lua "/data/$MAP_NAME"
    docker run --name osrm_backend-partition -t -v "$scriptDirectory/osrm_files:/data" "osrm/osrm-backend" osrm-partition "/data/$mapNameOsrm"
    docker run --name osrm_backend-customize -t -v "$scriptDirectory/osrm_files:/data" "osrm/osrm-backend" osrm-customize "/data/$mapNameOsrm"
    Write-Host "Creating container $CONTAINER_NAME"
    docker create --name $CONTAINER_NAME -p 5000:5000 -v "$scriptDirectory/osrm_files:/data" "osrm/osrm-backend" osrm-routed --algorithm mld "/data/$mapNameOsrm"
}
Write-Host "Starting container $CONTAINER_NAME"
docker start $CONTAINER_NAME