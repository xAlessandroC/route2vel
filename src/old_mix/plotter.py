import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd



if __name__ == "__main__":
    num_exp = 5

    data = []
    for i in range(1, num_exp+1):
        foldername = "test_cache_{}".format(i)

        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", foldername, "stats.txt"), "r") as f:
            file_data = f.readlines()

        data.append(file_data[0].strip().split(", "))

    print(data)
    df = pd.DataFrame(data, columns=["nodes", "distance", "osrm", "osm", "elevation", "interpolation"])
    df["distance"] = df["distance"].astype(float) 
    df["osrm"] = df["osrm"].astype(float)
    df["osm"] = df["osm"].astype(float)
    df["elevation"] = df["elevation"].astype(float)
    df["interpolation"] = df["interpolation"].astype(float)

    df["osrm"] = df["osrm"] * 1000
    df["osm"] = df["osm"] * 1000
    df["elevation"] = df["elevation"] * 1000
    df["interpolation"] = df["interpolation"] * 1000
    df['total_time'] = df['osrm'] + df['osm'] + df['elevation'] + df['interpolation']


    print(df.head())
    print(df.dtypes)

    ax = df.plot(x="nodes", y=["total_time", "osrm", "osm", "elevation", "interpolation"], kind="bar", rot=0, logy=True, edgecolor="black")
    ax.set_xticklabels([25, 50, 60, 70, 90])
    ax.set_xlabel("Number of route nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(10**0, 10**5)
    plt.legend(loc='center left', ncol = 5, bbox_to_anchor=(-0.1, 1.06), labels=["Total", "OSRM", "OSM", "Elevation", "Interpolation"])
    plt.title("System performance with caching system", y=1.09)
    plt.show()

    plt.savefig(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "charts", "cache.png"))
    plt.savefig(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "charts", "cache.pdf"), bbox_inches='tight')



    data = []
    for i in range(1, num_exp+1):
        foldername = "test_nocache_{}".format(i)

        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", foldername, "stats.txt"), "r") as f:
            file_data = f.readlines()

        data.append(file_data[0].strip().split(", "))

    print(data)
    df = pd.DataFrame(data, columns=["nodes", "distance", "osrm", "osm", "elevation", "interpolation"])

    df["distance"] = df["distance"].astype(float) 
    df["osrm"] = df["osrm"].astype(float)
    df["osm"] = df["osm"].astype(float)
    df["elevation"] = df["elevation"].astype(float)
    df["interpolation"] = df["interpolation"].astype(float)

    df["osrm"] = df["osrm"] * 1000
    df["osm"] = df["osm"] * 1000
    df["elevation"] = df["elevation"] * 1000
    df["interpolation"] = df["interpolation"] * 1000
    df['total_time'] = df['osrm'] + df['osm'] + df['elevation'] + df['interpolation']


    print(df.head())
    print(df.dtypes)

    ax = df.plot(x="nodes", y=["total_time", "osrm", "osm", "elevation", "interpolation"], kind="bar", rot=0, logy=True, edgecolor="black")
    ax.set_xticklabels([25, 50, 60, 70, 90])
    ax.set_xlabel("Number of route nodes")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(10**0, 10**5)
    plt.legend(loc='center left', ncol = 5, bbox_to_anchor=(-0.1, 1.06), labels=["Total", "OSRM", "OSM", "Elevation", "Interpolation"]) 
    plt.title("System performance without caching system", y=1.09)
    plt.show()

    plt.savefig(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "charts", "no_cache.png"))
    plt.savefig(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "charts", "no_cache.pdf"), bbox_inches='tight')

    