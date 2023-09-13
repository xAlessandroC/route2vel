# route2vel

Python package for extracting data about a route and interpolating points along its geometry for processing purposes.

(Made as part of thesis)

You can try a WebUI demo in the releases ([latest version](https://github.com/filloax/route2vel/releases/tag/0.7.0)), simply extract *Source Code.zip* and run either launch-webui.ps1 (Windows Powershell) or launch-webui.sh (Linux Bash). 
Running the WebUI requires [Docker](https://www.docker.com/) to be installed (to run the routing service [OSRM](https://project-osrm.org/), which will get downloaded by the launch script).
You can install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows or the `docker` package in most Linux distributions. The launch script will download and locally setup
all other dependencies.

The WebUI allows you to specify a starting and arrival location, and will compute the route. Then it will display it, and save the interpolated geometry data in a CSV at the path specified in the WebUI. The first launch
might take some time, as the program downloads map data.

---

#### Notes

Ported from private repo, which is why it was initialized already mostly made.