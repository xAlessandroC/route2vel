$containers = $(docker ps -aqf "name=osrm_backend")
docker kill $containers
docker rm $containers

$images = $(docker images osrm/osrm-backend -aq)

docker rmi $images

$Root = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)

Remove-Item -Recurse -Force "$Root/osrm_files/**"

Write-Host "Cleaned up OSRM containers, images and leftover files!"