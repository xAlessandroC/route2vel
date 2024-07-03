import osmnx as ox
from osmnx.utils_graph import route_to_gdf as ox_route_to_gdf
import networkx as nx
from .utils import logdebug
from typing import Any, Callable
import geopandas as gpd
from dataclasses import dataclass, field
from shapely import LineString, Point
import routingpy
import time
import os
from pathlib import Path

from .classes import RichDirection
from .loading import load_graph
from .interp import add_elevation_to_gdf_geometry

def find_route_osrm(
        route_steps: list[str|tuple[float,float]],
        route_profile = 'car',
        ox_graph: nx.MultiDiGraph = None, load_graph_name: str = None,
        load_graph = False,
        router = routingpy.OSRM(base_url='http://localhost:5000'),
        **loadgraphargs
        ) -> "RichDirection":
    """Find a route using a OSRM instance, either local (default) or from an arbitrary routingpy router endpoint.

    Args:
        route_steps (list[str | tuple[float,float]]): List of route steps, coordinates (long, lat) or names (can mix).
        route_profile (str): profile to run the routing on, like 'car', 'truck', etc. Check OSRM for possible options.
        ox_graph (nx.MultiDiGraph, optional): graph to use in the output object. Must set either this or load_graph_name.
        load_graph_name (str, optional): Name for the graph file that will be generated as cache when the output graph is 
            generated. Must set either this or ox_graph.
        load_graph (bool, optional): Immedialy generate the graph for the output route.
        router (optional): *routingpy* router to run directions with. Defaults to `routingpy.OSRM(base_url='http://localhost:5000')`.
    """
    route_coords = [ox.geocode(loc)[::-1] if type(loc) == str else loc for loc in route_steps]
    start = time.time()
    osrm_route = RichDirection(router.directions(route_coords, profile=route_profile, annotations=True), ox_graph, load_graph_name)
    end = time.time()
    with open(os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "tasks", "temp.txt"), "w") as f:
        print(f"{end - start}", file=f)

    if load_graph:
        osrm_route.get_graph(True, **loadgraphargs)

    return osrm_route

def route_to_gdf(
        graph: nx.MultiDiGraph, node_ids: list[int], weight: str = "length",
        node_columns: list[str] = ["elevation"], node_gdf: gpd.GeoDataFrame = None,
        ) -> gpd.GeoDataFrame:
    """Improved version of `ox.utils_graph.route_to_gdf`, allows to add
    node columns.
    Will return a GeoDataFrame with additional columns for each string in node_columns, 
    named as u<colname>, v<colname> for u and v, indexed by route order.
    """
    if node_gdf is None:
        node_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    for colname in node_columns:
        assert colname in node_gdf.columns

    gdf: gpd.GeoDataFrame = ox_route_to_gdf(graph, node_ids, weight).reset_index()
    for colname in node_columns:
        gdf[f"u_{colname}"] = gdf["u"].apply(lambda u: node_gdf.loc[u, colname])
        gdf[f"v_{colname}"] = gdf["v"].apply(lambda v: node_gdf.loc[v, colname])

    return gdf

def find_route(
        graph: nx.MultiDiGraph, loc1:str|tuple[float], loc2:str|tuple[float], 
        cost_function:str|Callable='prefer_main_roads',
        weight:str=None, 
        **extraargs
        ):
    """Find a route, using a cost function or a weight.
    Simpler pathfinding, for better behavior that depends on external values use OSRM with routingpy.

    Args:
        graph (nx.MultiDiGraph): Streets graph.  
        loc1 (str): Location name to geocode for origin.  
        loc2 (str): Location name to geocode for destination.  
        cost_function (str|Callable, optional): Function to use for cost comparision. Pass either a function that takes 
            (graph, edge data, start node id, end node id) as parameters, or name or a default function (use 
            `get_route_cost_functions` to get a list). Defaults to 'prefer_main_roads'. Unused if weight is set.  
        cost_key (str): Where the results of cost_function (if used) are stored. Defaults to 'cost'.  
        delete_cost_key_after (bool, optional): Remove computed cost after. Cleaner, but slightly affects performance due to cleanup. Defaults to True.  
        weight (str, optional): If set, use shortest path comparing this attribute in edges, otherwise use cost_function. Defaults to None.  
        **extraargs: passed to the cost function

    Returns:
        list[int]: List of node ids of the shortest route.
    """
    global _route_cost_functions

    use_func = weight is None
    func_to_use = _route_cost_functions[cost_function] if type(cost_function) == str else cost_function

    origin_coordinates = ox.geocode(loc1) if type(loc1) == str else loc1
    destination_coordinates = ox.geocode(loc2) if type(loc2) == str else loc2

    # Invert lat/long, osm uses lat as y and long as x
    origin_node = ox.nearest_nodes(graph, *(origin_coordinates[::-1]))
    destination_node = ox.nearest_nodes(graph, *(destination_coordinates[::-1]))

    logdebug("Origin node:", origin_node)
    logdebug("Destination node:", destination_node)

    logdebug(f"Using weight key: {weight}, cost_function: {cost_function if use_func else '-'}")

    shortest_route = ox.shortest_path(graph, origin_node, destination_node, weight=weight) \
        if not use_func else ox.shortest_path(graph, origin_node, destination_node, weight=lambda u, v, edge: func_to_use(graph, edge, u, v, **extraargs))

    return shortest_route

def get_route_cost_functions():
    global _route_cost_functions
    return list(_route_cost_functions.keys())

def get_route_cost_fun(key: str):
    global _route_cost_functions
    return _route_cost_functions[key]

@dataclass
class RouteNode:
    id: Any
    coords: tuple[float] #long then lat
    elevation: float|None = None
    # Underlying data of the node, containing usually OSM data
    data: dict = field(default_factory=dict)
    
    @property
    def long(self) -> float: return self.coords[0]
    @property
    def lat(self) -> float: return self.coords[1]
    # long, lat, elevation
    @property
    def coords3d(self) -> tuple[float]: return (*self.coords, self.elevation)
    @property
    def geometry(self) -> Point:
        if not self.__geometry:
            self.__geometry = Point(self.coords)
        return self.__geometry     
    
    def __getitem__(self, key):
        return getattr(self, key, self.data[key])
    
    def get(self, key, default):
        return getattr(self, key, self.data.get(key, default))
    
    @staticmethod
    def from_osmnx(data, osmid=None, long_key = 'x', lat_key = 'y', height_key = 'elevation'):
        return RouteNode(
            id=osmid if osmid else data['osmid'], coords=(data[long_key], data[lat_key]),
            elevation=data.get(height_key, None),
            data=data,
            )
    
@dataclass
class RouteEdge:
    id: Any
    length: float
    highway: str # osm highway data type (road type, etc)
    geometry: LineString
    speed_kph: float
    travel_time: float
    name: str = None
    grade: float = None
    grade_abs: float = None
    
    # Underlying data of the edge, containing usually OSM data
    data: dict = field(default_factory=dict)
        
    def __getitem__(self, key):
        return getattr(self, key, self.data[key])
    
    def get(self, key, default):
        return getattr(self, key, self.data.get(key, default))

    @staticmethod
    def from_osmnx(data):
        return RouteEdge(
            id=data['osmid'], name=data.get('name', None), length=float(data['length']), highway=data['highway'],
            geometry=data['geometry'], speed_kph=int(float(data['speed_kph'])), travel_time=float(data['travel_time']),
            grade=data.get('grade', None), grade_abs=data.get('grade_abs', None),
            data=data
            )
    
@dataclass
class RouteSegment:
    from_node: RouteNode
    to_node: RouteNode
    edge: RouteEdge

def route_to_edges(
        graph: nx.MultiDiGraph, 
        route: list[int]|list[tuple[float]],
        coords_are_lat_long = True,
    ) -> list[RouteSegment]:
    """Convert a route obtained with osmnx in list-of-nodeids format into a list of
    dicts containing start node, end node, and edge data of each edge in the route.
    Note: osmnx now includes ox.utils_graph.route_to_gdf(graph, node_ids, weight)
    which returns a GeoDataFrame for similar functionality; it doesn't include nodes, though.
    See `route.route_to_gdf` for an improved version.

    Args:
        graph (nx.MultiDiGraph): Graph to grab data from.
        route (list[int]|list[tuple[float]]): Route as list of graph node ids or coordinates (as lat/long tuples).
        coords_are_lat_long (optional, bool): Coords in route are treated as latitude first, longitude after. Defaults to True.

    Returns:
        list[RouteSegment]: List of route data
    """
    # sample edge structure: {'osmid': 529047868, 'name': 'Viale del Risorgimento', 'highway': 'residential', 
    # 'oneway': False, 'reversed': True, 'length': 6.443, 
    # 'geometry': <LINESTRING (11.328 44.488, 11.328 44.488, 11.328 44.488)>, 'speed_kph': 30.6, 
    # 'travel_time': 0.8}
    
    route_edges = []

    is_coords = type(route[0]) == tuple
    
    # ox nearest nodes takes long then lat
    long_idx = 1 if coords_are_lat_long else 0
    lat_idx = 0 if coords_are_lat_long else 1
    route_ids = ox.nearest_nodes(
        graph,
        [coords[long_idx] for coords in route],
        [coords[lat_idx] for coords in route],
    ) if is_coords else route

    for (i, prev_node), node in zip(enumerate(route_ids), route_ids[1:]):
        if prev_node == node:
            print(f"[WARN] Route has following identical nodes with id {node}, were {route[i]} and {route[i + 1]}, too close coordinates for map?")
            continue
        
        # MultiDiGraph: more than one edge
        edges = graph[prev_node][node]
        if (len(edges) > 1):
            logdebug(f"\tLink {prev_node}-{node} >1 edges!", [edges[e].get("name", "nan") for e in edges])
        # choose best by travel time
        edge = edges[min(edges, key=lambda e: edges[e]['travel_time'])]
        route_edges.append(RouteSegment(
            from_node=RouteNode.from_osmnx(graph.nodes[prev_node], osmid=prev_node),
            to_node=RouteNode.from_osmnx(graph.nodes[node], osmid=node),
            edge=RouteEdge.from_osmnx(edge),
        ))

    return route_edges

def measure_route(route: list[RouteSegment]|gpd.GeoDataFrame, weight: str, zeroIfMissing:bool=True):
    """Sum a weight of all edges in a route.

    Args:
        route (list[RouteSegment]): Route as returned by `route.route_to_edges`.
        weight (str): Either field name or key in `RouteEdge.data`.
        zeroIfMissing (bool, optional): Consider a key to be 0 if missing, otherwise error. Defaults to True. Ignored with geodataframes.

    Returns:
        float: Sum of the weight.
    """
    if type(route) == gpd.GeoDataFrame:
        return route[weight].sum()
    elif zeroIfMissing:
        return sum(map(lambda entry: getattr(entry.edge, weight, entry.edge.data.get(weight, 0)), route))
    else:
        return sum(map(lambda entry: getattr(entry.edge, weight, entry.edge.data[weight]), route)) # will error if missing weight

# can take 1 or more, instead of only 1 or only more
def plot_graph_routes(graph, routes: list[list[int]], route_colors="r", route_linewidths=4, node_size=0, **pg_kwargs):
    if len(routes) > 1:
        return ox.plot_graph_routes(graph, routes, route_colors, route_linewidths, node_size=0, **pg_kwargs)
    elif len(routes) == 1:
        route_colors = route_colors if type(route_colors) == str else route_colors[0] 
        route_linewidths = route_linewidths if type(route_linewidths) in [int, float] else route_linewidths[0] 
        return ox.plot_graph_route(graph, routes[0], route_color=route_colors, route_linewidth=route_linewidths, node_size=0, **pg_kwargs)
    
def df_from_route(route: list[dict]):
    return gpd.GeoDataFrame([{"from": r["from_id"], "to": r["to_id"], **r["edge"]} for r in route], geometry="geometry")
    
_truck_road_priority = {
    'motorway': 0.85,
    'trunk': 0.85,
    'primary': 1,
    'secondary': 2,
    'tertiary ': 2,
    'unclassified': 10,
    'residential': 10,
    '_default': 10,
}

def _route_cost_truck_default(
        graph: nx.MultiDiGraph, edge_data: dict, edge_from: int, edge_to: int, 
        # extra args
        basic_weight: str = 'travel_time', penalty_exp: int = 1, priority_table:'dict[str,float]'=_truck_road_priority
    ):
    best_edge: dict = min(edge_data.values(), key=lambda e: e[basic_weight]) 

    cost_basic = best_edge[basic_weight]
    hw = best_edge['highway']
    penalty = 1
    reward = 1

    if type(hw) == str:    
        # X_link highway types are motorway/autobahn/etc links, only replace if not found
        penalty = priority_table.get(hw, priority_table.get(hw.replace('_link', ''), priority_table['_default']))
    else:
        # On multiple values https://wiki.openstreetmap.org/wiki/Multiple_values
        # We will use the highest penalty, assuming multiple highways are
        # a combination of multiple street datas
        penalty = max(priority_table.get(x, priority_table.get(x.replace('_link', ''), priority_table['_default'])) for x in hw)

    return cost_basic * (penalty ** penalty_exp) / reward

_route_cost_functions = {
    # Penalizes city streets, can take basic_weight to specify weight to use for reference (default: 'travel_time')
    'prefer_main_roads': _route_cost_truck_default,
}
