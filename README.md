# route2vel

Python package for extracting data about a route and interpolating points along its geometry for processing purposes.

(Made as part of thesis)

You can try a WebUI demo in the releases ([latest version](https://github.com/filloax/route2vel/releases/latest)), simply extract *Source Code.zip* and run either *launch-webui.ps1* (Windows Powershell) or *launch-webui.sh* (Linux Bash). 
Running the WebUI requires [Docker](https://www.docker.com/) and Python 3 to be installed (to run the routing service [OSRM](https://project-osrm.org/), which will get downloaded by the launch script). Make sure to install the `venv` module with `pip install venv` if it's not already installed.
You can install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows or the `docker` package in most Linux distributions. The launch script will download and locally setup
all other dependencies.

Note that it requires approximately 3GB of free space!

The WebUI allows you to specify a starting and arrival location, and will compute the route. 
Then it will display it, and save the interpolated geometry data in a CSV at the 
path specified in the WebUI. The first launch
will take some time, as the program downloads map data.

---

#### Notes

- Ported from private repo, which is why it was initialized already mostly made.
- If there are issues during the install, try also deleting the created docker containers (4 containers starting with OSRM) before restarting. You can easily do this using the *cleanup-osrm.ps1* (.sh is yet todo) to do this more quickly.

---

#### Possible additions

Ideas on possible improvements that are not implemented (yet).
- Directly using the local .osm.pbf file to load graphs instead of the Overpass API, now that osmnx supports it. Would help in both avoiding API calls and having OSRM and graph data always in sync.
- Locally host Open Topo Data (as seen [in its doc](https://www.opentopodata.org/)) instead of using its free API. Might not be necessary depending on the usecase, in case a paid API is able to be used.