from flask import Flask, render_template, request, jsonify
import webbrowser
from flask_socketio import SocketIO
from matplotlib.figure import Figure
import route2vel
from route2vel.postprocess import calc_curvature, interp_gdf_to_csv
import os
import re
import osmnx as ox
import io
from matplotlib import pyplot as plt
import matplotlib
from geopandas import GeoDataFrame
import contextily as cx

app = Flask(__name__)
socketio = SocketIO(app)

root_dir = "."
if os.path.dirname(__file__).endswith('src'):
    root_dir = '..'

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_calc')
def on_start_calc(data):
    start_location = data['start_location']
    end_location = data['end_location']
    sampling_distance = int(data.get('sampling_distance', 5))
    csv_path = data.get('csv_path', 'route_output_full.csv')
    run_path_calc(start_location, end_location, sampling_distance, csv_path)

def run_path_calc(start_location: str, end_location: str, sampling_distance: int, csv_path: str):
    graph_name = re.sub(r'[^\w_. -]', '_', f"{start_location.lower().strip()}-{end_location.lower().strip()}")
        
    print(f"Finding route from {start_location} to {end_location}")
    
    socketio.emit('path_update', {
        'start': True,
    })
    
    route_dir = route2vel.find_route_osrm([start_location, end_location], load_graph=True, load_graph_name=graph_name)

    socketio.emit('path_result', {
        'length': route_dir.distance,
        'duration': route_dir.duration,
        'coords': len(route_dir.geometry),
        'nodes': len(route_dir.nodelist()),
    })
    
    print("Calculated route, starting interpolation...")
    
    interp_args = {}
    if sampling_distance is not None:
        interp_args['distance_treshold'] = sampling_distance
    interp_dir = route2vel.interp_from_route(route_dir, **interp_args)
    
    socketio.emit('interp_gdf', {
        'table': interp_dir.split_gdf.head().to_html()
    })
    
    route2vel.utils.debug = True
    sampled_gdf = interp_dir.get_points_with_density(
        sampling_distance, 
        return_gdf=True, 
        in_meters=True, 
        gdf_columns=['base_idx', 'junction', 'speed_kph'],
    )
    
    calc_curvature(sampled_gdf)
    
    csv_path = os.path.abspath(csv_path)

    interp_gdf_to_csv(
        sampled_gdf, csv_path, 
        separate_roundabout=True, 
        add_tract_start_col=True, 
        extra_cols=['speed_kph', 'curvature'],
    )

    socketio.emit('saved_csv', {
        'path': csv_path
    })

    print("Sending images...")
    socketio.emit('route_image', {
        'started': True,
    })
    route_img = generate_image(route_dir.gdf)
    print("Generated route image")
    split_img = generate_image(interp_dir.split_gdf)
    print("Generated split image")
    sample_img = generate_image(sampled_gdf)
    print("Generated sample image")
    socketio.emit('route_image', {
        'route_img': route_img,
        'split_img': split_img,
        'sample_img': sample_img,
    })
    print("Sent images")

def generate_image(gdf: GeoDataFrame):
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(25, 15))
    gdf.plot(ax=ax)
    cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    return buffer.read()

def init_cx_cache():
    cx_cache_path = os.path.join(route2vel.cfg['resources_dir'], 'contextily_cache')
    cx.set_cache_dir(cx_cache_path)
    # delete old stuff

def main():
    route2vel.load_config(root_dir)
    init_cx_cache()
    matplotlib.use('agg')
    debug = False
    port = 8080
    if not debug:
        webbrowser.open(f'http://localhost:{port}')
    socketio.run(app, debug=debug, port=port)

if __name__ == '__main__':
    main()