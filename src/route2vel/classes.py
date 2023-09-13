from routingpy.direction import Direction
from geopandas import GeoDataFrame
import numpy as np
import networkx as nx
from shapely import Point, LineString
import sys
from pyproj import Transformer, CRS

from .utils import get_utm_zone, logdebug

try:
    from matplotlib import pyplot as plt
    import contextily as cx
except ImportError:
    pass

_needed_modules_later = ["interp", "route", "loading"]

# avoid circular imports
def _init():
    global split_route_gdf_by_curves, get_point_in_route_gdf
    global route_to_gdf, add_elevation_to_gdf_geometry
    global load_graph

    from .interp import split_route_gdf_by_curves, get_point_in_route_gdf
    from .route import route_to_gdf, add_elevation_to_gdf_geometry
    from .loading import load_graph

# Wrapper for direction, intended for main usecase
class RichDirection(Direction):
    graph: nx.MultiDiGraph
    gdf: GeoDataFrame
    _graph_name: str

    def __init__(self, base_direction:Direction, graph: nx.MultiDiGraph = None, graph_filename: str = None):
        super().__init__(base_direction.geometry, base_direction.duration, base_direction.distance, base_direction._raw)
        self.gdf = None
        self.graph = graph
        self._graph_name = graph_filename

        if self.graph is None and graph_filename is None:
            raise ValueError("Creating direction without either graph information or a graph name, would lead to redownloading the graph each time")

    def nodelist(self) -> list[int]:
        """Get a list of OSM ids the route passes through. Tested with routes returned by OSRM with annotations=True.
        If the object has an osmnx graph, keep only node ids in the graph (so, removing simplified ones).

        Returns:
            list[int]: OSM node ids.
        """
        node_ids = []
        for route_leg in self._raw['routes'][0]['legs']:
            node_ids.extend([int(id) for id in route_leg['annotation']['nodes']])

        if self.graph:
            node_ids = list(filter(lambda id: id in self.graph.nodes, node_ids))
            # print(f"Removed {len(list(filter(lambda id: id not in self.graph.nodes, node_ids)))}")

        return node_ids
    
    def calc_gdf(self, recalc=True, rich_elevation=True, **routegdfargs) -> bool:
        assert self.graph is not None

        if self.gdf is None or recalc:
            self.gdf = route_to_gdf(self.graph, self.nodelist(), **routegdfargs)
            if rich_elevation:
                self.gdf = add_elevation_to_gdf_geometry(self.gdf)
            return True
        return False

    def get_gdf(self, 
                weight: str = "length", 
                node_columns: list[str] = [], node_gdf: GeoDataFrame = None,
                recalc=False,
                ) -> GeoDataFrame:
        assert self.graph is not None

        self.calc_gdf(recalc, weight=weight, node_columns=node_columns, node_gdf=node_gdf)

        return self.gdf
    
    def bounds(self, flat=False, margin=0.0005) -> tuple[tuple[float, float], tuple[float, float]]|tuple[float, float, float, float]:
        """Returns bounds of geometry as bottom left, top right (lon, lat), or NSEW tuple"""
        minx, miny, maxx, maxy = None, None, None, None
        # geometry is lon, lat
        for pt in self.geometry:
            minx = min(pt[0], minx) if minx else pt[0]
            miny = min(pt[1], miny) if miny else pt[1]
            maxx = max(pt[0], maxx) if maxx else pt[0]
            maxy = max(pt[1], maxy) if maxy else pt[1]
        minx -= margin 
        miny -= margin 
        maxx += margin 
        maxy += margin 
        # latitude increases towards north pole
        if flat:
            return (maxy, miny, maxx, minx)
        else:
            return ((minx, miny), (maxx, maxy))
    
    def get_graph(self, replace_own=False, cache_graph=None, **loadargs) -> nx.MultiDiGraph:
        if cache_graph is None:
            cache_graph = self._graph_name is not None

        graph = load_graph(self.bounds(True), filename=self._graph_name, cache_graph=cache_graph, **loadargs)
        if self.graph is None or replace_own:
            self.graph = graph
        return graph
    
    def plot(self, show=True, title="OSRM route"):
        assert plt
        assert cx

        self.calc_gdf(recalc=False)

        crs = 'epsg:4326'
        points_gdf = GeoDataFrame({'geometry': [Point(x) for x in self.geometry]}, crs=crs)
        fig, ax = plt.subplots(figsize=(12, 12))
        self.gdf.plot(ax=ax, color='k')
        points_gdf.plot(ax = ax, color='r', edgecolor='k')
        cx.add_basemap(ax, crs=crs)
        plt.title(title)
        if show:
            plt.show()
        return fig


class InterpolatingDirection(Direction):
    base: RichDirection
    split_gdf: GeoDataFrame

    def __init__(self, base_direction: RichDirection, distance_treshold: float, use_elevation=True, **splitargs):
        super().__init__(base_direction.geometry, base_direction.duration, base_direction.distance, base_direction._raw)

        base_direction.calc_gdf(recalc=False, rich_elevation=use_elevation)

        self.base = base_direction
        has_2d_column = "geometry2d" in self.base.gdf
        if has_2d_column:
            self.split_gdf = split_route_gdf_by_curves(self.base.gdf, distance_treshold, geom_key="geometry2d", secondary_geom_keys=["geometry"], **splitargs)
        else:
            self.split_gdf = split_route_gdf_by_curves(self.base.gdf, distance_treshold, **splitargs)

    def get_points(
            self, t: float|np.ndarray, 
            in_meters: bool = False, meters_utm_zone: int = None, return_meters_crs: bool = False,
            return_gdf = False, gdf_unwrap_points = False, gdf_columns: list[str] = [],
            **getpointargs
        ) -> np.ndarray|tuple[np.ndarray, list[bool]]:
        """Get points as per the `get_point_in_route_gdf` and `get_point_in_linestrings_select_curves`
        functions, optionally projecting to meters.

        Args:
            t (float | np.ndarray): Route pct (or array of) to get the poitns at.
            in_meters (bool, optional): Return coordinates in meters. Defaults to False.
            meters_utm_zone (int, optional): [https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system](UTM zone)
                to use to project to meters. Defaults to None (which automatically determines it from route points).
            return_meters_crs (bool, optional): adds to the return tuple the CRS used for projecting to meters
            return_gdf (bool, optional): return output as GeoDataFrame (will be tuple if return_metres_crs is True, 
                with gdf as first arg). By default, the table contains the points in the geometry column, and 
                defaults the getpointargs return_label and return_linestring_indices to True to include them in the 
                'is_curve' and 'split_gdf_index' columns.
            gdf_unwrap_points (bool, optional): return points in the gdf as 'lon', 'lat', 'ele' columns with floats
                additionally to a single 'geometry' column with shapely Points as default
            gdf_columns (list[str], optional): Columns to include in the returned gdf from the original gdf, to have additional
                OSM data. Will automatically set return_linestring_indices to True to be able to match th elinestring.
                
            keyword args: args for `get_point_in_route_gdf` and `get_point_in_linestrings_select_curves`
        """
        if return_gdf:
            if 'return_label' not in getpointargs:
                getpointargs['return_label'] = True
            if 'return_linestring_indices' not in getpointargs:
                getpointargs['return_linestring_indices'] = True
            if not getpointargs['return_linestring_indices'] and gdf_columns:
                getpointargs['return_linestring_indices'] = True

        return_label = getpointargs.get('return_label', False)
        return_linestring_indices = getpointargs.get('return_linestring_indices', False)
        get_points_output = get_point_in_route_gdf(self.split_gdf, t, **getpointargs)
        if return_label or return_linestring_indices:
            get_points_output = list(get_points_output)
            points: np.ndarray = get_points_output.pop(0)
            if return_label:
                labels: list[bool] = get_points_output.pop(0)
            if return_linestring_indices:
                linestring_indices: list[int] = get_points_output.pop(0)
        else:
            points: np.ndarray = get_points_output
                        
        if in_meters:
            if meters_utm_zone is None:
                points_average = np.array([line.centroid.coords[0] for line in self.split_gdf.geometry]).mean(axis=0)
                meters_utm_zone = get_utm_zone(points_average[0], points_average[1])
                logdebug(f"Projecting to UTM zone {meters_utm_zone} (auto, from point {points_average})")
            else:
                logdebug(f"Projecting to UTM zone {meters_utm_zone}")
           
            input_crs: CRS = self.base.gdf.crs
            target_crs = CRS(proj='utm', zone=meters_utm_zone, units="m", ellipsis=input_crs.datum.ellipsoid)
            transformer = Transformer.from_crs(input_crs, target_crs, always_xy=True)
            points = np.array([np.array((*transformer.transform(p[0], p[1]), *p[2:])) for p in points])
                        
        if not return_gdf:
            output = (points,)
            if return_label:
                output += (labels,)
            if return_linestring_indices:
                output += (linestring_indices,)
        else:
            crs = self.base.gdf.crs
            if in_meters:
                crs = target_crs
            gdf = GeoDataFrame({
                'geometry': [Point(pt) for pt in points],
            }, geometry='geometry', crs=crs)
            if gdf_unwrap_points:
                points_t = points.transpose()
                gdf['lon'] = points_t[0]
                gdf['lat'] = points_t[1]
                gdf['ele'] = points_t[2]

            if return_label:
                gdf['is_curve'] = labels
            if return_linestring_indices:
                gdf['split_gdf_index'] = linestring_indices

            if len(gdf_columns) > 0:
                gdf = gdf.merge(self.split_gdf[gdf_columns], left_on='split_gdf_index', right_index=True)

            output = (gdf,)

        if return_meters_crs:
            output += (target_crs,)

        if len(output) == 1:
            output = output[0]
            
        return output
    
    def get_points_with_density(self, point_dist_meters: float, **getpointargs):
        return self.get_points_with_num(self.pt_num_for_density(point_dist_meters), **getpointargs)
    
    def get_points_with_num(self, num: int, **getpointargs):
        return self.get_points(np.linspace(0, 1, num=num), **getpointargs)
    
    def pt_num_for_density(self, point_dist_meters: float):
        return round(self.distance / point_dist_meters)

    def get_gdf_to_plot(self, num_points=None, geom_key=None, as_lines=False, **getpointargs):
        if num_points is None:
            num_points = self.pt_num_for_density(1)
        if geom_key is None:
            geom_key="geometry2d" if "geometry2d" in self.split_gdf else "geometry"
        pts, pt_labels = self.get_points(np.linspace(0, 1, num=num_points), 
                                         geom_key=geom_key, 
                                         return_label=True,
                                         **getpointargs
                                         )
        return GeoDataFrame({
            'geometry': [LineString([pt, pt2]) for pt, pt2 in zip(pts[:-1], pts[1:])] if as_lines 
                else [Point(pt) for pt in pts],
            'curve': ['curve' if l else 'straight' for l in (pt_labels[:-1] if as_lines else pt_labels)],
        }, geometry='geometry', crs=self.split_gdf.crs)

    def plot(self, pts_per_edge=None, show=True, title="OSRM route"):
        assert plt
        assert cx

        gdf_to_plot = self.get_gdf_to_plot(pts_per_edge)

        crs = 'epsg:4326'
        points_gdf = GeoDataFrame({'geometry': [Point(x) for x in self.geometry]}, crs=crs)
        fig, ax = plt.subplots(figsize=(12, 12))
        gdf_to_plot.to_crs(crs).plot(ax=ax, column='curve', categorical=True, legend=True, cmap='RdYlGn_r', s=1)
        points_gdf.plot(ax = ax, color='k', edgecolor='k')
        cx.add_basemap(ax, crs=crs)
        plt.title(title)
        if show:
            plt.show()

if all(map(lambda module: f"route2vel.{module}" in sys.modules, _needed_modules_later)):
    _init()