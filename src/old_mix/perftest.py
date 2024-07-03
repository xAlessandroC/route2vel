import re
import sys
import os
import io
import time
import shutil
import socketio
import argparse
import traceback
import route2vel
import traceback
import subprocess
import matlab.engine
import contextily as cx
import matplotlib.pyplot as plt
from shapely import Point, LineString


from pathlib import Path
from geopandas import GeoDataFrame
from route2vel.postprocess import calc_curvature, interp_gdf_to_csv


# parser = argparse.ArgumentParser(description="Route2Vel is a tool to find routes between points and extract velocity profiles")
# parser.add_argument("--start", metavar=("lat", "lon"), type=float, n2, required=True, help="Starting location (lat, lon)")
# parser.add_argument("--end", metavar=("lat", "lon"), type=float, n2, required=True, help="Ending location (lat, lon)")
# parser.add_argument("--intermediate", metavar=("lat", "lon"), type=float, n2, action="append", help="Intermediate location (lat, lon) - 0 or more")
# parser.add_argument("--sampling", metavar="d", type=int, default=5, help="Sampling distance in meters")
# parser.add_argument("--websocket", metavar="host", type=str, default="http://localhost:8080", help="Websocket host to send updates to")
# parser.add_argument("--websocket-room", metavar="room", type=str, help="Websocket room to send updates to")
# parser.add_argument("--output-dir", metavar="dir", type=str, default=r"C:\Users\alexc\Desktop\Universita\PhD\Projects\route2vel\tasks\default", help="Output directory for csv and images")


def generate_image(gdf: GeoDataFrame, save_path: str):
    gdf = gdf.to_crs(epsg=3857)
    # gdf = gdf.set_geometry("geometry2d")
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf.plot(ax=ax, linewidth=4, color="red")
    cx.add_basemap(ax, attribution_size=0, source=cx.providers.OpenStreetMap.Mapnik)
    fig.savefig(buffer, format='png', bbox_inches='tight')
    fig.savefig(save_path, format='png', bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()


def run_engine(start, end, intermediate, sampling, output_dir):
    # Check output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Stats vars
        osrm_phase_start = -1,
        osrm_phase_end = -1
        elevation_phase_start = -1
        elevation_phase_end = -1
        interpolation_phase_start = -1
        interpolation_phase_end = -1
        acceleration_phase_start = -1
        acceleration_phase_end = -1

        # Find route
        start_location_formatted = "{},{}".format(start[0], start[1])
        end_location_formatted = "{},{}".format(end[0], end[1])
        start_location = (start[1], start[0])
        end_location = (end[1], end[0])
        intermediate_locations = [(p[1], p[0]) for p in intermediate] if intermediate else []

        graph_name = re.sub(r'[^\w_. -]', '_', f"{start_location_formatted.lower().strip()}-{end_location_formatted.lower().strip()}")
        print(f"Finding route from {start_location_formatted} to {end_location_formatted}")

        osrm_phase_start = time.time()
        route_dir = route2vel.find_route_osrm([start_location, *intermediate_locations, end_location], load_graph=True, load_graph_name=graph_name)
        osrm_phase_end = time.time()

        interpolation_phase_start = time.time()
        interp_dir = route2vel.interp_from_route(route_dir)
        print("Route found: {}".format(route_dir.geometry))

        route2vel.utils.debug = True
        sampled_gdf = interp_dir.get_points_with_density(
            sampling,
            return_gdf=True,
            in_meters=True, #XXX: should be set to true in order to be compliant with DIN postprocessing
            gdf_columns=['base_idx', 'junction', 'speed_kph'] if 'junction' in interp_dir.split_gdf.columns else ['base_idx','speed_kph'] #XXX: fix juncture missing,
        )
        interpolation_phase = time.time()
        calc_curvature(sampled_gdf)
        sampled_points = [[sh_point.x, sh_point.y, sh_point.z] for sh_point in sampled_gdf["geometry"].tolist()]
        latlon_sampled_points = [[sh_point.x, sh_point.y, sh_point.z] for sh_point in sampled_gdf["latlong_coords"].tolist()]

        csv_path = os.path.join(output_dir, "route_output_full.csv")

        interp_gdf_to_csv(
            sampled_gdf, csv_path, 
            separate_roundabout=True if 'junction' in interp_dir.split_gdf.columns else False,  #XXX: fix juncture missing, 
            add_tract_start_col=True, 
            extra_cols=['speed_kph', 'curvature'],
        )

        route_img = generate_image(route_dir.gdf, os.path.join(output_dir, "route.png"))
        split_img = generate_image(GeoDataFrame({'geometry': [Point(pt) for pt in route_dir.geometry]}, geometry='geometry', crs=route_dir.gdf.crs), os.path.join(output_dir, "nodelist.png"))
        sample_img = generate_image(sampled_gdf, os.path.join(output_dir, "sampled_route.png"))
    except Exception as e:
        time.sleep(1)
        print(traceback.format_exc(), file=sys.stderr)

    try:
        with open(os.path.join(output_dir, "param"), "w") as param_file:
            print(f"Init coord: {start_location}", file=param_file)
            print(f"End coord: {end_location}", file=param_file)
            print(f"Intermediate: {intermediate_locations}", file=param_file)
            print(f"Sampling frequency: {sampling}", file=param_file)
    except Exception as e:
        print("Error saving parameter file")

    try:
        nodelist = route_dir.nodelist()
        node_ids = []
        for route_leg in route_dir._raw['routes'][0]['legs']:
            node_ids.extend([int(id) for id in route_leg['annotation']['nodes']])
        with open(os.path.join(output_dir, "debug"), "w") as param_file:
            print(f"Nodes OSRM: {node_ids}", file=param_file)
            print(f"Nodes OSM (Filtered): {nodelist}", file=param_file)
            print(f"Comparison:\n {nodelist[:-1]}\n{nodelist[1:]}", file=param_file)
            for edge in route_dir.graph.edges(keys=True):
                print("{}".format(edge),file=param_file)
    except Exception as e:
        print("Error saving debug file")

    try:
        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","temp.txt"), "r") as f:
            content = f.readlines()
        with open(os.path.join(output_dir, "stats.txt"), "w") as stats_file:
            print("{}, {}, {}, {}, {}, {}".format(len(nodelist), route_dir.distance, content[0].strip(), content[1].strip(), content[2].strip(), interpolation_phase - interpolation_phase_start), file=stats_file)
            print(f"N NODES: {len(nodelist)}", file=stats_file)
            print(f"DISTANCE: {route_dir.distance}", file=stats_file)
            print(f"OSRM: {content[0].strip()}", file=stats_file)
            print(f"OSM: {content[1].strip()}", file=stats_file)
            print(f"Elevation: {content[2].strip()}", file=stats_file)
            print(f"Interpolation: {interpolation_phase - interpolation_phase_start}", file=stats_file)
            # print(f"Acceleration: {acceleration_phase_end - acceleration_phase_start}", file=stats_file)
    except Exception as e:
        print("Error saving stats file", e)


if __name__ == "__main__":
    # Namespace(start=[44.48479526325984, 11.279909312725069], end=[44.48088237098375, 11.282100677490234], intermediate=[[44.485391480260574, 11.277090311050415], [44.4823678084413, 11.27547025680542]], sampling=10, websocket='http://localhost:8080', websocket_room='aR5g1vIblkwCkaGWAAAz', output_dir='/home/routevel/route_vel_service/route2vel/tasks/1714123235842')
    
    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])

    run_engine(
        (44.48479526325984, 11.279909312725069),
        (44.48088237098375, 11.282100677490234),
        [(44.485391480260574, 11.277090311050415), (44.4823678084413, 11.27547025680542)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_1")
    )

    run_engine(
        (44.48479526325984, 11.279909312725069),
        (44.48088237098375, 11.282100677490234),
        [(44.485391480260574, 11.277090311050415), (44.4823678084413, 11.27547025680542)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_1")
    )

    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])

    # 45 nodes

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.492579101786454, 11.297807693481445),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_2")
    )

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.492579101786454, 11.297807693481445),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_2")
    )

    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])

    
    # 60 nodes
    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.49482910251596, 11.327413916587831),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_3")
    )

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.49482910251596, 11.327413916587831),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_3")
    )

    # 70 nodes
    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.49588638285721, 11.339864730834961),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_4")
    )

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.49588638285721, 11.339864730834961),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_4")
    )

    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])

    # 90 nodes
    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.48780041329098, 11.354627609252931),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)], 
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_5")
    )

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.48780041329098, 11.354627609252931),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)], 
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_5")
    )

    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])

    # 100 nodes
    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.48094081218434, 11.374025344848635),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_nocache_6")
    )

    run_engine(
        (44.48088237098375, 11.282100677490234),
        (44.48094081218434, 11.374025344848635),
        [(44.482895741979675, 11.276103258132936), (44.48567152962956, 11.277261972427368)],
        10,
        os.path.join(Path(os.path.abspath(__file__)).parent.parent,"tasks","test_cache_6")
    )

    subprocess.run(["/home/routevel/route_vel_service/route2vel/clear_cache.sh"])