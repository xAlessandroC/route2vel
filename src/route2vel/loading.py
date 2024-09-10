import osmnx as ox
import networkx as nx
import os, sys, time
from shapely.geometry import LineString, MultiLineString
from shapely import linestrings, unary_union
from shapely.ops import linemerge
import pandas as pd
import geopandas as gpd
from .config import cfg
from .utils import logdebug, extract_connections_sequence
import pickle
from warnings import warn
import numpy as np
from tqdm import tqdm
from pathlib import Path

# used in rework on add elevation function
from osmnx.elevation import _downloader, InsufficientResponseError, _elevation_request

ROAD_MAXSPEED = {
    'it': {
        'unclassified': 70,
        'residential': 50,
        'primary': 90,
        'secondary': 90,
        'tertiary': 90,
        'trunk': 110,
        'motorway': 130,
    },
}

def load_graph(area: str|tuple[float], 
               cache_graph=True, cache_graph_pickle=True, filename: str = None, 
               allow_merge_different_osmids = True,
               **downloadargs
               ) -> nx.MultiDiGraph:
    """Load a graph representing a road network.

    Args:
        area (str | tuple[float]): Area name or tuple of 4 coordinates (N, S, E, W) to use as bounding box.
        cache_graph (bool, optional): Load from filename if present or download and save there if not. 
            If False, always download. Defaults to True.
        filename (str, optional): Filename to use. If None, extrapolate from area name (if area is a string). Defaults to None.
        **downloadargs: args passed to `download_graph`.

    Returns:
        nx.MultiDiGraph: Graph of the road network.
    """
    
    if type(area) != str and cache_graph and not filename:
        raise Exception("If you use a non-string area and want to save/load from file, you must give a filename!")
    
    fn = _get_path(_filename_from_area(filename if filename else area), with_pickle=cache_graph_pickle, ox_strict_simplify=allow_merge_different_osmids)
    if os.path.exists(fn) and cache_graph:
        print(f"Loading graph from {os.path.abspath(fn)} ...")
        time1 = time.time()
        if cache_graph_pickle:
            with open(fn, 'rb') as f:
                graph = pickle.load(f)
        else:
            graph = ox.load_graphml(fn)
        time2 = time.time()
        print(f"Loaded in {time2 - time1:.2f}s!")
        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", "temp.txt"), "a") as f:
            print(f"{time2 - time1}", file=f)
            print(f"{time2 - time1}", file=f)
        return graph
    else:
        return download_graph(area, 
                              save=cache_graph, filename=filename, save_with_pickle=cache_graph_pickle, 
                              allow_merge_different_osmids=allow_merge_different_osmids,
                              **downloadargs
                              )

def download_graph(
    area: str|tuple[float], fill_data=True, save=True, filename: str = None,
    save_with_pickle=True,
    simplify=True, allow_merge_different_osmids = True,
    keep_raw_copy=True,
    network_type='drive',
    **fillargs
    ) -> nx.MultiDiGraph:
    """Download a graph from OSM.

    Args:
        area (str | tuple[float]): Area name or tuple of 4 coordinates (N, S, E, W) to use as bounding box.
        fill_data (bool, optional): Fill missing data, like default road max speeds, elevation, geometry. Defaults to True.
        save (bool, optional): Save graph to file. Defaults to True.
        filename (str, optional): Graph filename. If None and area is string, extrapolate from there. Defaults to None.
        keep_raw_copy: Also save another copy of the graph as downloaded, in case another call of load with different parameters on the same area is done later.

    Returns:
        nx.MultiDiGraph: Downloaded graph.
    """
    if type(area) != str and (save or keep_raw_copy) and not filename:
        raise Exception("If you use a non-string area and want to save/load from file, you must give a filename!")

    graph = None

    if type(area) == str or filename:
        raw_fn = _get_path(_filename_from_area(filename if filename else area), with_pickle=save_with_pickle, raw=True)
        if os.path.exists(raw_fn):
            if save_with_pickle:
                with open(raw_fn, 'rb') as f:
                    graph = pickle.load(f)
            else:
                graph = ox.load_graphml(raw_fn)
            print(f"Loaded raw graph from {raw_fn} as previously downloaded!")
    else:
        raw_fn = None

    if not graph:
        # Set OSM filter explicitly to handle specific road types - we do not want to include service roads
        osm_filters = (
            f'["highway"]["area"!~"yes"]'
            # f'["access"!~"private"]'
            f'["highway"!~"abandoned|bridleway|bus_guideway|corridor|cycleway|elevator|'
            f"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|"
            f'steps|track"]'
            f'["motor_vehicle"!~"no"]["motorcar"!~"no"]'
            # f'["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
        )

        # osm_filters = (
        #     f'["access"!~"no|agricultural|forestry|emergency|psv|customers|private|delivery|destination"]'
        #     f'["highway"]["access"!~"private"]'
        #     f'["service"!~"emergency_access"]'
        #     f'["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|'
        #     f"escalator|footway|no|path|pedestrian|planned|platform|proposed|raceway|razed|"
        #     f'steps|track"]'
        # )

        # Create the graph of the area from OSM data. It will download the data and create the graph
        print(f"Downloading graph for {area} ...")
        time1 = time.time()
        graph = ox.graph_from_place(area, network_type=network_type, simplify=False, custom_filter=osm_filters) if type(area) == str \
            else ox.graph_from_bbox(*area, network_type=network_type, simplify=False, custom_filter=osm_filters)
        time2 = time.time()
        print(f"Downloaded OSM map in {time2 - time1}s!")
        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", "temp.txt"), "a") as f:
            print(f"{time2 - time1}", file=f)

        if keep_raw_copy:
            if save_with_pickle:
                os.makedirs(os.path.dirname(raw_fn), exist_ok=True)
                with open(raw_fn, 'wb') as f:
                    pickle.dump(graph, f)
            else:
                ox.save_graphml(graph, raw_fn)
            print(f"Saved raw copy to {raw_fn}!")

    if simplify:
        print(f"Simplifying... (strict={allow_merge_different_osmids})")
        graph = ox.simplify_graph(graph, strict=allow_merge_different_osmids) # avoid having multiple osmids in same edge
        print(f"Simplified!")

    if fill_data:
        graph = fill_graph_data(graph, **fillargs)

    if save:
        fn = _get_path(_filename_from_area(filename if filename else area), with_pickle=save_with_pickle, ox_strict_simplify=allow_merge_different_osmids)
        _create_dir_if_missing(fn)
        if save_with_pickle:
            with open(fn, 'wb') as f:
                pickle.dump(graph, f)
        else:
            ox.save_graphml(graph, fn)
        print(f"Saved to {os.path.abspath(fn)}")

    return graph

# Data filling

def fill_graph_data(
            graph, return_info=False, 
            geometry=True, speed=True, time=True,
            elevation=True, ele_method=None, ele_api_key=None,
            grades=True, abs_grades=True,
            hwy_speeds=ROAD_MAXSPEED['it']
        ):
    if time and not speed:
        print("[WARN] Time data needs speed!", file=sys.stderr)
    if grades and not elevation:
        print("[WARN] Grade data needs elevation!", file=sys.stderr)

    # add_edge_speeds can take a fallback table for missing maxspeed
    # with default values
    (graph, missing_edges) = fill_geometry(graph, return_missing=True) if geometry else (graph, None)

    if elevation:
        import time as tm
        start = tm.time()
        graph = add_elevation(graph, ele_method, ele_api_key)
        end = tm.time()
        print(f"Added elevation in {end - start}s")
        with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", "temp.txt"), "a") as f:
            print(f"{end - start}", file=f)
        if grades:
            graph = ox.add_edge_grades(graph, abs_grades)
            logdebug(f"Added edge grades")
    
    if speed:
        graph = ox.add_edge_speeds(graph, hwy_speeds=hwy_speeds)
        logdebug(f"Added edge speeds")

    if time and speed:
        graph = ox.add_edge_travel_times(graph)
        logdebug(f"Added edge travel times")

    if return_info:
        return graph, {
            "missing_geometry": missing_edges,
        }
    return graph

def fill_geometry(graph: nx.MultiDiGraph, return_missing=False):
    out_graph = graph.copy()
    had_missing_idxs = []
    for u, v, k, attr in out_graph.edges(data=True, keys=True):
        # node for reference: {'y': 44.4879522, 'x': 11.3281188, 'street_count': 1}
        if "geometry" not in attr:
            node1 = out_graph.nodes[u]
            node2 = out_graph.nodes[v]
            line = LineString([(node1['x'], node1['y']), (node2['x'], node2['y'])])
            attr["geometry"] = line
            had_missing_idxs.append((u, v, k))
    logdebug(f"Filled geometry. Added geometry in {len(had_missing_idxs)} edges")

    if return_missing:
        return out_graph, had_missing_idxs
    return out_graph

VALID_METHODS = [
    'google',
    'opentopodata',
]

def add_elevation(graph: nx.MultiDiGraph, method:str=None, api_key:str=None):
    """Add elevation data to graph

    Args:
        graph (nx.MultiDiGraph): Graph to add elevation to.
        method (str, optional): Will use `config` if undefined.
        api_key (str, optional):  Will use `config` if undefined.
    """
    if method is None: method = cfg['ele_method']
    if api_key is None: api_key = cfg['ele_api_key']
    if method not in VALID_METHODS: 
        raise ValueError(f"Invalid method {method}, valid ones are {VALID_METHODS}")
    
    print("Adding elevation...")
    t1 = time.time()
    try:
        if method == 'google':
            graph = _add_node_elevations_google(
                graph, 
                api_key=api_key,
            )
        elif method == 'opentopodata':
            # api key is unused here
            # As seen [here](https://www.opentopodata.org/#public-api), the
            # open topo data Public API is restricted to 1call/second, 100 locations/call,
            # 1000 calls/day
            # COMMENT: OPENTOPODATA API (change to local one with cache?)
            graph = _add_node_elevations_google(
                graph, 
                url_template='http://localhost:5005/v1/test-dataset?locations={}', #XXX: was 'https://api.opentopodata.org/v1/aster30m?locations={}&key={}'
                max_locations_per_batch=1000, #XXX: was 100
                pause=0.1, #XXX: was 1
                api_key=api_key,
            )
    except KeyError:
        raise Exception("Elevation API response contained error!")
    t2 = time.time()
    print(f"Added elevation in {t2 - t1:.2f}s")
    return graph
   
def _add_node_elevations_google(
    G,
    api_key=None,
    max_locations_per_batch=350,
    pause=0,
    precision=None,
    url_template="https://maps.googleapis.com/maps/api/elevation/json?locations={}&key={}",
    # route2vel: added by us
    use_tqdm=True,
):  # pragma: no cover
    """
    Modified version of the osmnx method to show a progress bar with tqdm, as this is usually the long part.

    Original pydoc below.
    ---
    Add `elevation` (meters) attribute to each node using a web service.

    By default, this uses the Google Maps Elevation API but you can optionally
    use an equivalent API with the same interface and response format, such as
    Open Topo Data. The Google Maps Elevation API requires an API key but
    other providers may not.

    For a free local alternative see the `add_node_elevations_raster`
    function. See also the `add_edge_grades` function.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    api_key : string
        a valid API key, can be None if the API does not require a key
    max_locations_per_batch : int
        max number of coordinate pairs to submit in each API call (if this is
        too high, the server will reject the request because its character
        limit exceeds the max allowed)
    pause : float
        time to pause between API calls, which can be increased if you get
        rate limited
    precision : int
        deprecated, do not use
    url_template : string
        a URL string template for the API endpoint, containing exactly two
        parameters: `locations` and `key`; for example, for Open Topo Data:
        `"https://api.opentopodata.org/v1/aster30m?locations={}&key={}"`

    Returns
    -------
    G : networkx.MultiDiGraph
        graph with node elevation attributes
    """
    if precision is None:
        precision = 3
    else:
        warn(
            "the `precision` parameter is deprecated and will be removed in a future release",
            stacklevel=2,
        )

    # make a pandas series of all the nodes' coordinates as 'lat,lng'
    # round coordinates to 5 decimal places (approx 1 meter) to be able to fit
    # in more locations per API call
    node_points = pd.Series(
        {node: f'{data["y"]:.5f},{data["x"]:.5f}' for node, data in G.nodes(data=True)}
    )
    n_calls = int(np.ceil(len(node_points) / max_locations_per_batch))
    domain = _downloader._hostname_from_url(url_template)
    ox.utils.log(f"Requesting node elevations from {domain!r} in {n_calls} request(s)")

    # break the series of coordinates into chunks of size max_locations_per_batch
    # API format is locations=lat,lng|lat,lng|lat,lng|lat,lng...
    results = []

    node_range = range(0, len(node_points), max_locations_per_batch)
    if use_tqdm:
        node_range = tqdm(node_range)

    for i in node_range:
        chunk = node_points.iloc[i : i + max_locations_per_batch]
        locations = "|".join(chunk)
        url = url_template.format(locations, api_key)

        # download and append these elevation results to list of all results
        response_json = _elevation_request(url, pause)
        results.extend(response_json["results"])

    # sanity check that all our vectors have the same number of elements
    msg = f"Graph has {len(G):,} nodes and we received {len(results):,} results from {domain!r}"
    ox.utils.log(msg)
    if not (len(results) == len(G) == len(node_points)):
        err_msg = f"{msg}\n{response_json}"
        raise InsufficientResponseError(err_msg)

    # add elevation as an attribute to the nodes
    df = pd.DataFrame(node_points, columns=["node_points"])
    df["elevation"] = [result["elevation"] for result in results]
    df["elevation"] = df["elevation"].round(precision)
    nx.set_node_attributes(G, name="elevation", values=df["elevation"].to_dict())
    ox.utils.log(f"Added elevation data from {domain!r} to all nodes.")

    return G

WAY_EQUAL_ATTRS = ['oneway', 'lanes', 'ref', 'name', 'highway', 'maxspeed', 'bridge', 'access', 'junction', 'service', 'tunnel', 'width']

def compress_edges_dataframe(edges_df: gpd.GeoDataFrame, nodes_df: gpd.GeoDataFrame = None, return_geometry=True, drop_complex_shapes=False) -> gpd.GeoDataFrame:
    """Takes in a graph dataframe as generated by osmnx containing OSM data, returns a dataframe simplified by
    merging edges with the same osmid and generating geometry that connects their nodes.
    This doesn't also merge edges without a crossing inbetween, like osmnx.simplify_graph does.
    This does make the gdf not a graph anymore, as it merges edges that had a crossing inbetween.

    Args:
        edges_df (gpd.GeoDataFrame): Edge GeoDataframe as returned by osmnx.graph_to_gdfs, from a non-simplified graph.
        nodes_df (gpd.GeoDataFrame): Node GeoDataframe as returned by osmnx.graph_to_gdfs, from a non-simplified graph. Not necessary if return_geometry=False.
        return_geometry (bool): Add a geometry column with Linestrings to the new dataframe, and return it as a GeoDataFrame. Needs nodes_df to be set.
        drop_complex_shapes (bool): Avoid calculating complex shapes (multi-loops, etc), for quick data analysis purposes.

    Returns:
        gpd.GeoDataFrame: Merged dataframe, indexed by osmid.
    """

    if return_geometry and nodes_df is None:
        raise ValueError("Needs nodes df to return geometry!")
    
    # Group the edges by 'osmid' and aggregate the other attributes
    grouped_df = edges_df[~(edges_df.reversed)].reset_index().groupby('osmid').agg({
        **{
            key: 'first' for key in WAY_EQUAL_ATTRS
        },
        'length': 'sum',
        'geometry': lambda x: {line.coords[0]: line for line in x if line is not None}, #in case merging a non-simplified dataframe
        'u': lambda x: x.tolist(),
        'v': lambda x: x.tolist(),
    }).reset_index()

    grouped_df['u_all'] = grouped_df.u
    grouped_df['v_all'] = grouped_df.v
    grouped_df['geometry_all'] = grouped_df.geometry
    grouped_df['nodes'] = grouped_df.apply(lambda row: _get_row_node_seq(row, drop_complex_shapes=drop_complex_shapes), axis=1)
    dropped_rows = grouped_df[grouped_df.nodes.isnull()]
    grouped_df = grouped_df[grouped_df.nodes.notnull()]
    grouped_df['u'] = grouped_df.apply(lambda row: row.nodes[0], axis=1)
    grouped_df['v'] = grouped_df.apply(lambda row: row.nodes[-1], axis=1)
    if return_geometry:
        grouped_df['nodes_line'] = grouped_df.nodes.apply(lambda x: _node_osmids_to_linestring(x, nodes_df))
        grouped_df = grouped_df.assign(geometry = grouped_df.apply(lambda row: _merge_row_geometries(row), axis=1))
    
    if len(dropped_rows) > 0:
        print(f"Dropped {len(dropped_rows)} rows")

    if return_geometry:
        return gpd.GeoDataFrame(
                grouped_df.drop(['u_all', 'v_all'], axis=1), 
                geometry='geometry',
                crs = edges_df.crs,
            ).set_index(['osmid'])
    else:
        return pd.DataFrame(
                grouped_df.drop(['u_all', 'v_all'], axis=1), 
            ).set_index(['osmid'])

def _get_row_node_seq(row: pd.Series, ku='u_all', kv='v_all', drop_complex_shapes=False) -> list[int]:
    global dbg
    dbg = row
    out = extract_connections_sequence(row[ku], row[kv])
    if len(out) < len(row[ku]):
        if drop_complex_shapes:
            return None
        else:
            mismatch_count = len(pd.Series(row[ku] + row[kv]).unique()) - len(row[ku])
            raise Exception(f"Row with id {row.osmid} name {row.name} has unconnected edges! (Found {len(out)}, need {len(row[ku])}, mismatch count {mismatch_count})")
    return out
    
def _node_osmids_to_linestring(osmids: list[int], nodes_df: pd.DataFrame) -> LineString:
    points = nodes_df.loc[osmids][['x', 'y']]
    return linestrings(points.x, points.y)

def _merge_row_geometries(row: pd.Series) -> LineString:
    if row['geometry_all']:
        all_geom: dict = row['geometry_all']
        segments_ordered = []
        for node_start_coord, next_coord in zip(row['nodes_line'].coords[:-1], row['nodes_line'].coords[1:]):
            segment = all_geom.get(node_start_coord, None)
            if segment:
                segments_ordered.append(segment)
            else:
                segments_ordered.append(LineString((node_start_coord, next_coord)))
                print(f"[WARN] Merging geometries, match not found for node at {node_start_coord}")
        union = unary_union(segments_ordered)
        return _merge_multilinestring(union)
    else:
        return row['nodes_line']

def _merge_multilinestring(shape: LineString|MultiLineString):
    if type(shape) is MultiLineString:
        return LineString([coord for line in shape.geoms for coord in line.coords])
    else:
        return shape

def _filename_from_area(area: str):
    return f"{area.split(',')[0].strip()}"

def _get_path(name: str, with_pickle=True, ox_strict_simplify=True, raw=False):
    fname = name
    if raw:
        fname += "_raw"
    else:
        if not ox_strict_simplify:
            fname += "_nomerge"
    fname += f".{'graphnx.pickle' if with_pickle else 'graphml'}"

    return os.path.abspath(os.path.join(cfg['graphs_dir'], fname))

def _create_dir_if_missing(filepath: str):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.makedirs(dir)
