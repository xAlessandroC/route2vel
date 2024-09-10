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
import matlab.engine
import contextily as cx
import matplotlib.pyplot as plt
from shapely import Point, LineString


from pathlib import Path
from geopandas import GeoDataFrame
from route2vel.postprocess import calc_curvature, interp_gdf_to_csv


parser = argparse.ArgumentParser(description="Route2Vel is a tool to find routes between points and extract velocity profiles")
parser.add_argument("--start", metavar="startAddr", type=str, required=True, help="Star address")
parser.add_argument("--end", metavar="endAddr", type=str, required=True, help="End address")
parser.add_argument("--intermediate", metavar="address", type=str, action="append", help="Intermediate address  - 0 or more")
parser.add_argument("--sampling", metavar="d", type=int, default=5, help="Sampling distance in meters")
parser.add_argument("--websocket", metavar="host", type=str, default="http://localhost:8080", help="Websocket host to send updates to")
parser.add_argument("--websocket-room", metavar="room", type=str, help="Websocket room to send updates to")
parser.add_argument("--output-dir", metavar="dir", type=str, default=r"C:\Users\alexc\Desktop\Universita\PhD\Projects\route2vel\tasks\default", help="Output directory for csv and images")


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
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()

def generate_eleimage(gdf: GeoDataFrame, save_path: str):
    print(gdf.head())
    gdf['elevation'] = gdf['geometry'].apply(lambda point: point.z)
    gdf = gdf.to_crs(epsg=3857)
    # gdf = gdf.set_geometry("geometry2d")
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    gdf.plot(ax=ax, linewidth=4, column='elevation', cmap='viridis', legend=True)
    cx.add_basemap(ax, attribution_size=0, source=cx.providers.OpenStreetMap.Mapnik)
    fig.savefig(buffer, format='png', bbox_inches='tight')
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()

#XXX: osrm was capable of handling addresses also.. this dedicated script may not be necessary
if __name__ == "__main__":

    args = parser.parse_args()
    print(args)    
    # Check output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Connect to websocket
    ws_client = socketio.SimpleClient()
    ws_client.connect(args.websocket)

    # Emit message so that the server can join the client to the room
    ws_client.emit("join", {
        "room": args.websocket_room
    })
    time.sleep(1)

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
        start_location_formatted = args.start
        end_location_formatted = args.end
        start_location = args.start
        end_location = args.end
        intermediate_locations = [p for p in args.intermediate] if args.intermediate else []

        graph_name = re.sub(r'[^\w_. -]', '_', f"{start_location_formatted.lower().strip()}-{end_location_formatted.lower().strip()}")
        print(f"Finding route from {start_location_formatted} to {end_location_formatted}")

        # socketio.emit('path_update', {
        #     'start': True,
        # })

        ws_client.emit("update_by_addr", {
            "message": f"Ricerca percorso...",
            "room": args.websocket_room
        })
        time.sleep(1)

        osrm_phase_start = time.time()
        route_dir = route2vel.find_route_osrm([start_location, *intermediate_locations, end_location], load_graph=True, load_graph_name=graph_name)
        osrm_phase_end = time.time()
        # print("Route found: {}".format(route_dir))
        # print("Nodelist: {}".format(route_dir.nodelist()))
        # total_points = [start_location, *intermediate_locations, end_location]
        # for start, end in zip(total_points[:-1], total_points[1:]):
        #     route_dir = route2vel.find_route_osrm([start, end], load_graph=True, load_graph_name=graph_name)
        #     print("Route found: {}".format(route_dir))
        #     print("Nodelist: {}".format(route_dir.nodelist()))

        ws_client.emit("update_by_addr", {
            "message": "Interpolazione percorso...",
            "room": args.websocket_room
        })
        elevation_phase_start = time.time()
        interp_dir = route2vel.interp_from_route(route_dir)
        elevation_phase_end = time.time()
        print("Route found: {}".format(route_dir.geometry))

        route2vel.utils.debug = True
        interpolation_phase_start = time.time()
        sampled_gdf = interp_dir.get_points_with_density(
            args.sampling,
            return_gdf=True,
            in_meters=True, #XXX: should be set to true in order to be compliant with DIN postprocessing
            gdf_columns=['base_idx', 'junction', 'speed_kph'] if 'junction' in interp_dir.split_gdf.columns else ['base_idx','speed_kph'] #XXX: fix juncture missing,
        )
        interpolation_phase = time.time()
        calc_curvature(sampled_gdf)
        sampled_points = [[sh_point.x, sh_point.y, sh_point.z] for sh_point in sampled_gdf["geometry"].tolist()]
        latlon_sampled_points = [[sh_point.x, sh_point.y, sh_point.z] for sh_point in sampled_gdf["latlong_coords"].tolist()]

        csv_path = os.path.join(args.output_dir, "route_output_full.csv")

        ws_client.emit("update_by_addr", {
            "message": "Generazione CSV...",
            "room": args.websocket_room
        })
        # time.sleep(1)

        interp_gdf_to_csv(
            sampled_gdf, csv_path, 
            separate_roundabout=True if 'junction' in interp_dir.split_gdf.columns else False,  #XXX: fix juncture missing, 
            add_tract_start_col=True, 
            extra_cols=['speed_kph', 'curvature'],
        )

        ws_client.emit("update_by_addr", {
            "message": "Generazione immagini...",
            "room": args.websocket_room
        })
        # time.sleep(1)

        # route_img = generate_image(route_dir.gdf, os.path.join(args.output_dir, "route.pdf"))
        # split_img = generate_image(GeoDataFrame({'geometry': [Point(pt) for pt in route_dir.geometry]}, geometry='geometry', crs=route_dir.gdf.crs), os.path.join(args.output_dir, "nodelist.pdf"))
        # sample_img = generate_image(sampled_gdf, os.path.join(args.output_dir, "sampled_route.pdf"))
        # sample_eleimg = generate_eleimage(sampled_gdf, os.path.join(args.output_dir, "sampled_eleroute.pdf"))


        #XXX: add matlab invocation
        try:
            acceleration_phase_start = time.time()
            test_csv_path = os.path.join(Path(__file__).parent.parent, "din_pipeline", "route_output_full.csv")
            # shutil.copy(test_csv_path, args.output_dir)
            eng = matlab.engine.start_matlab("-nodesktop -nosplash -nodisplay")
            src_din = eng.genpath(os.path.join(Path(__file__).parent.parent, "din_pipeline"))
            eng.addpath(src_din, nargout=0)
            eng.acc2Twin(matlab.double(1.00), matlab.double(1.10), matlab.double(1.10), matlab.double(1.05), matlab.double(1.00), csv_path)
            eng.close()
            acceleration_phase_end = time.time()
        except Exception as e:
            print("Error in matlab invocation: ", traceback.format_exc())
            with open(os.path.join(args.output_dir, "matlab_stacktrace.txt"), "w") as stacktrace_file:
                print(f"{traceback.format_exc()}", file=stacktrace_file)
        #XXX: end din_pipeline

        ## Send all the results
        ws_client.emit("route_data_by_addr", {
            "type": "route",
            "length": route_dir.distance,
            "duration": route_dir.duration,
            "coords": route_dir.geometry,
            "geojson": route_dir.gdf.drop("geometry2d", axis=1).to_json(),
            "room": args.websocket_room
        })

        ws_client.emit("route_data_by_addr", {
            "type": "sampling",
            "coords": latlon_sampled_points,
            "room": args.websocket_room
        })

        # Send all the generated svgs
        # XXX: armir-ale timeout send
        for svg_file in os.listdir(args.output_dir):
            if svg_file.endswith(".svg"):
                print("Sending {}".format(svg_file))
                with open(os.path.join(args.output_dir, svg_file), "r") as svg_f:
                    ws_client.emit("route_data_by_addr", {
                        "type": "image",
                        "name": svg_file,
                        "data": svg_f.read(),
                        "room": args.websocket_room
                    })
                    time.sleep(1)
        time.sleep(10)

        # Send all the identified wc datasets
        # XXX: armir-ale timeout send
        for wc_file in os.listdir(args.output_dir):
            if wc_file.endswith(".csv"):
                print("Sending {}".format(wc_file))
                with open(os.path.join(args.output_dir, wc_file), "r") as wc_f:
                    ws_client.emit("route_wc_data_by_addr", {
                        "type": "image",
                        "name": wc_file,
                        "data": wc_f.read(),
                        "room": args.websocket_room
                    })
                    time.sleep(1)
        time.sleep(10)

        ws_client.disconnect()
    except Exception as e:
        time.sleep(1)
        print(traceback.format_exc(), file=sys.stderr)
        ws_client.emit("route_error_by_addr", {
            "message": "Errore nella ricerca del percorso...",
            "room": args.websocket_room
        })
        time.sleep(1)
        ws_client.disconnect()

    try:
        with open(os.path.join(args.output_dir, "param.txt"), "w") as param_file:
            print(f"Init coord: {start_location}", file=param_file)
            print(f"End coord: {end_location}", file=param_file)
            print(f"Intermediate: {intermediate_locations}", file=param_file)
            print(f"Sampling frequency: {args.sampling}", file=param_file)
    except Exception as e:
        print("Error saving parameter file")

    try:
        nodelist = route_dir.nodelist()
        node_ids = []
        for route_leg in route_dir._raw['routes'][0]['legs']:
            node_ids.extend([int(id) for id in route_leg['annotation']['nodes']])
        with open(os.path.join(args.output_dir, "debug.txt"), "w") as param_file:
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
        with open(os.path.join(args.output_dir, "stats.txt"), "w") as stats_file:
            print("{}, {}, {}, {}, {}, {}".format(len(nodelist), route_dir.distance, content[0].strip(), content[1].strip(), content[2].strip(), interpolation_phase - interpolation_phase_start), file=stats_file)
            print(f"N NODES: {len(nodelist)}", file=stats_file)
            print(f"DISTANCE: {route_dir.distance}", file=stats_file)
            print(f"OSRM: {content[0].strip()}", file=stats_file)
            print(f"OSM: {content[1].strip()}", file=stats_file)
            print(f"Elevation: {content[2].strip()}", file=stats_file)
            print(f"Interpolation: {interpolation_phase - interpolation_phase_start}", file=stats_file)
            # print(f"Acceleration: {acceleration_phase_end - acceleration_phase_start}", file=stats_file)
    except Exception as e:
        print("Error saving stats file")
