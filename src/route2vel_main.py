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

from pathlib import Path
from geopandas import GeoDataFrame
from route2vel.postprocess import calc_curvature, interp_gdf_to_csv


parser = argparse.ArgumentParser(description="Route2Vel is a tool to find routes between points and extract velocity profiles")
parser.add_argument("--start", metavar=("lat", "lon"), type=float, nargs=2, required=True, help="Starting location (lat, lon)")
parser.add_argument("--end", metavar=("lat", "lon"), type=float, nargs=2, required=True, help="Ending location (lat, lon)")
parser.add_argument("--intermediate", metavar=("lat", "lon"), type=float, nargs=2, action="append", help="Intermediate location (lat, lon) - 0 or more")
parser.add_argument("--sampling", metavar="d", type=int, default=5, help="Sampling distance in meters")
parser.add_argument("--websocket", metavar="host", type=str, default="http://localhost:8080", help="Websocket host to send updates to")
parser.add_argument("--websocket-room", metavar="room", type=str, help="Websocket room to send updates to")
parser.add_argument("--output-dir", metavar="dir", type=str, default=r"C:\Users\alexc\Desktop\Universita\PhD\Projects\route2vel\tasks\default", help="Output directory for csv and images")


def generate_image(gdf: GeoDataFrame, save_path: str):
    gdf = gdf.to_crs(epsg=3857)
    # gdf = gdf.set_geometry("geometry2d")
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(25, 15))
    gdf.plot(ax=ax)
    cx.add_basemap(ax, attribution_size=0, source=cx.providers.OpenStreetMap.Mapnik)
    fig.savefig(buffer, format='png', bbox_inches='tight')
    fig.savefig(save_path, format='png', bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()


if __name__ == "__main__":
    args = parser.parse_args()

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

        # Find route
        start_location_formatted = "{},{}".format(args.start[0], args.start[1])
        end_location_formatted = "{},{}".format(args.end[0], args.end[1])
        start_location = (args.start[1], args.start[0])
        end_location = (args.end[1], args.end[0])
        intermediate_locations = [(p[1], p[0]) for p in args.intermediate] if args.intermediate else []

        graph_name = re.sub(r'[^\w_. -]', '_', f"{start_location_formatted.lower().strip()}-{end_location_formatted.lower().strip()}")
        print(f"Finding route from {start_location_formatted} to {end_location_formatted}")

        # socketio.emit('path_update', {
        #     'start': True,
        # })

        ws_client.emit("update", {
            "message": "Ricerca percorso...",
            "room": args.websocket_room
        })
        time.sleep(1)

        route_dir = route2vel.find_route_osrm([start_location, *intermediate_locations, end_location], load_graph=True, load_graph_name=graph_name)

        ws_client.emit("update", {
            "message": "Interpolazione percorso...",
            "room": args.websocket_room
        })
        interp_dir = route2vel.interp_from_route(route_dir)
        print("Route found: {}".format(route_dir.gdf.head()))

        route2vel.utils.debug = True
        sampled_gdf = interp_dir.get_points_with_density(
            args.sampling,
            return_gdf=True,
            in_meters=False,
            gdf_columns=['base_idx', 'junction', 'speed_kph'] if 'junction' in interp_dir.split_gdf.columns else ['base_idx','speed_kph'] #XXX: fix juncture missing,
        )
        calc_curvature(sampled_gdf)
        sampled_points = [[sh_point.x, sh_point.y, sh_point.z] for sh_point in sampled_gdf["geometry"].tolist()]

        csv_path = os.path.join(args.output_dir, "route_output_full.csv")

        ws_client.emit("update", {
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

        ws_client.emit("update", {
            "message": "Generazione immagini...",
            "room": args.websocket_room
        })
        # time.sleep(1)

        # route_img = generate_image(route_dir.gdf, os.path.join(args.output_dir, "route.png"))
        # split_img = generate_image(interp_dir.split_gdf, os.path.join(args.output_dir, "route_interp.png"))
        # sample_img = generate_image(sampled_gdf, os.path.join(args.output_dir, "sampled_route.png"))

        #XXX: add matlab invocation
        try:
            test_csv_path = os.path.join(Path(__file__).parent.parent, "din_pipeline", "route_output_full.csv")
            # shutil.copy(test_csv_path, args.output_dir)
            eng = matlab.engine.start_matlab("-nodesktop -nosplash -nodisplay")
            src_din = eng.genpath(os.path.join(Path(__file__).parent.parent, "din_pipeline"))
            eng.addpath(src_din, nargout=0)
            eng.acc2Twin(matlab.double(1.00), matlab.double(1.10), matlab.double(1.10), matlab.double(1.05), matlab.double(1.00), csv_path)
            eng.close()
        except Exception as e:
            print("Error in matlab invocation: ", traceback.format_exc())
        #XXX: end din_pipeline

        # time.sleep(5)
        ## Send all the results
        ws_client.emit("route_data", {
            "type": "route",
            "length": route_dir.distance,
            "duration": route_dir.duration,
            "coords": route_dir.geometry,
            "geojson": route_dir.gdf.drop("geometry2d", axis=1).to_json(),
            "room": args.websocket_room
        })

        ws_client.emit("route_data", {
            "type": "sampling",
            "coords": sampled_points,
            "room": args.websocket_room
        })

        # Send all the results svgs
        for svg_file in os.listdir(args.output_dir):
            if svg_file.endswith(".svg"):
                print("Sending {}".format(svg_file))
                with open(os.path.join(args.output_dir, svg_file), "r") as svg_f:
                    ws_client.emit("route_data", {
                        "type": "image",
                        "name": svg_file,
                        "data": svg_f.read(),
                        "room": args.websocket_room
                    })
                    time.sleep(1)

        ws_client.disconnect()
    except Exception as e:
        time.sleep(1)
        print(traceback.format_exc(), file=sys.stderr)
        ws_client.emit("route_error", {
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
