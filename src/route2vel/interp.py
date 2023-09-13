import networkx as nx
import numpy as np
from routingpy.direction import Direction
from shapely import LineString, Point
from shapely.ops import linemerge, transform
from pyproj import CRS, Transformer
import shapely
import scipy
import scipy.interpolate as scint
from typing import Callable, Tuple
import osmnx as ox
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
from pandas import Series
from types import SimpleNamespace

from .utils import get_segment_indices, get_utm_zone, logdebug
from .classes import InterpolatingDirection, RichDirection

def interp_from_route(direction: RichDirection, distance_treshold: float = 15, use_elevation=True, **splitargs):
    return InterpolatingDirection(direction, distance_treshold, use_elevation, **splitargs)

def split_route_gdf_by_curves(
    route_gdf: GeoDataFrame, distance_treshold: float, 
    geom_key="geometry", 
    secondary_geom_keys=["geometry2d"], 
    merge_short_segments=True, merge_short_args:dict|SimpleNamespace = {},
    **splitargs
) -> GeoDataFrame:
    """Split a gdf's rows representing a route's segments with a geometry column having linestrings, into more rows
    such that each row has points of the same distance density category (under or below the treshold); to be used
    to determine interpolation method.

    Args:
        route_gdf (GeoDataFrame): GeoDataFrame containing a column with LineStrings, such that each row's linestring starts where the previous one ends.
        distance_treshold (float): Distance in metres to use when splitting the linestrings by distance density.
        geom_key (str, optional): Main geometry column key, should contain linestrings. Defaults to "geometry".
        secondary_geom_keys (list, optional): If there are more geometry columns to split alongside the main one. 
            Should contain linestrings of the same length as the linestring in the main column. Defaults to ["geometry2d"].
        merge_short_segments (bool, optional): Merge short segments with `merge_short_segments`. Defaults to True.
        merge_short_args (dict or SimpleNamespace): Args to pass to `merge_short_segments`

    Returns:
        GeoDataFrame: New geodataframe with the split rows.
    """
    split_global_geometry_with_labels = pd.DataFrame(route_gdf.apply(lambda row: _split_gdf_row(row, distance_treshold, geom_key, merge_short_segments, merge_short_args, **splitargs), axis=1).tolist())
    graph_df_copy = pd.DataFrame(route_gdf)
    graph_df_copy[["segments", "iscurve"]] = split_global_geometry_with_labels

    for secondary_geom_key in secondary_geom_keys:
        if secondary_geom_key in route_gdf.columns:
            graph_df_copy[secondary_geom_key] = graph_df_copy.apply(
                lambda row: _split_secondary_geometry(row[secondary_geom_key], row["segments"]), axis=1
            )

    columns_to_split_on = ['segments', 'iscurve'] + [col for col in secondary_geom_keys if col in route_gdf.columns]

    split_route_gdf = gpd.GeoDataFrame(
        graph_df_copy.explode(columns_to_split_on).reset_index().drop([geom_key], axis=1)\
            .rename(columns={'index': 'base_idx'})\
            .rename(columns={'segments': geom_key}),
        geometry=geom_key,
        crs = route_gdf.crs,
    )
    return split_route_gdf

def _split_gdf_row(row: Series, distance_treshold: float, geom_key="geometry", do_merge_short_segments=True, merge_short_args={}, **splitargs) -> tuple[list[LineString], list[bool]]:
    geom = row[geom_key]
    linestrings, labels = split_linestring_by_curves(geom, distance_treshold, way_data=row, return_labels=True, return_geoseries=False, **splitargs)
    if do_merge_short_segments:
        linestrings, labels = merge_short_segments(linestrings, labels, **merge_short_args)
    return linestrings, labels

def _split_secondary_geometry(secondary_geometry: LineString, segments: list[LineString]) -> list[LineString]:
    # Split the secondary geometry based on the segments obtained from the main geometry
    segment_bounds = get_segment_indices(segments)
    return [LineString(secondary_geometry.coords[b[0]:b[1]]) for b in segment_bounds]

def coord_distance_meters(coords1:tuple, coords2:tuple):
    """Convert coord distance to meters. Make sure to use WGS-84 coordinates.
    """
    # Get first two values inverted, ignore eventual third values like height etc
    return ox.distance.great_circle_vec(*(coords1[1::-1]), *(coords2[1::-1]))

def split_linestring_by_curves(linestring: LineString, distance_treshold: float, 
                                distance_func: Callable[[tuple, tuple], float] = coord_distance_meters,
                                way_data: Series|dict = None,
                                lon_first=True,
                                add_borders_as_new_group_first=True,
                                return_labels=True,
                                return_geoseries=False,
                                ) -> "list[LineString|Point]|GeoSeries|tuple[list[LineString|Point]|GeoSeries,list[bool]]":
    """Split a linestring into a list of Linestrings (or Linestrings and Points, depending on arguments) 
    separating straight segments from curving ones, distinguishing based on density and other metadata.

    Args:
        linestring (shapely.Linestring): The linestring to split
        distance_treshold (float): The distance between points where they stop being close enough
            to be considered a curve tract.
        distance_func: a function that takes two tuples of coordinates (in lon, lat format) and returns
            the distance between them, in the same unit as `distance_treshold`. Defaults to using
            `ox.distance.great_circle_vec`, which uses meters.
        way_data: a dict or pandas Series containing OSM metadata of the edge this linestring is a part of, if any.
            Used for additional splitting conditions. Defaults to True.
        lon_first: if the input has coordinates in (lon, lat) instead of (lat, lon) format. Defaults to True.
        add_borders_as_new_group_first: If the last point of a segment should be the starting point of the next,
            if False segments will be disjointed, and segments consisting of a single Point could happen. 
            Defaults to True.
        return_labels: Also return a list containing if each segment is a curve or not, the same length as the
            normal result. Defaults to True.
        return_geoseries: Return the list of Linestrings as a geopandas Geoseries instead of a Python list.

    Returns:
        list[LineString|Point]|GeoSeries: The sequence of segments.
        list[bool]: Labels with True if the corresponding segment is a curve, or False otherwise.
            Only if `return_labels` is True.
    """

    result = []
    labels = []
    normalized_coords = [coord[::-1] if not lon_first else coord for coord in linestring.coords]
    from_normal = (lambda x: x) if lon_first else (lambda x: x[::-1])
    current_group = [linestring.coords[0]]
    group_is_curve = None
    for prev_point_n, current_point_n in zip(normalized_coords[:-1], normalized_coords[1:]):
        cur_is_curve = None

        if way_data is not None and cur_is_curve is None:
            cur_is_curve = _way_data_is_curve(way_data)

        if cur_is_curve is None:
            distance = distance_func(prev_point_n, current_point_n)
            cur_is_curve = distance <= distance_treshold

        if group_is_curve is None:
            group_is_curve = cur_is_curve

        if cur_is_curve == group_is_curve:
            current_group.append(from_normal(current_point_n))
        else:
            if len(current_group) > 1:
                result.append(LineString(current_group))
            else:
                result.append(Point(current_group[0]))
            labels.append(group_is_curve)

            group_is_curve = cur_is_curve
            if add_borders_as_new_group_first:
                current_group = [from_normal(prev_point_n), from_normal(current_point_n)]
            else:
                current_group = [from_normal(current_point_n)]
    
    if len(current_group) > 1:
        result.append(LineString(current_group))
    else:
        result.append(Point(current_group[0]))
    labels.append(group_is_curve)

    if return_geoseries:
        result = GeoSeries(result)

    if return_labels:
        return result, labels
    else:
        return result
    
def merge_short_segments(
    linestrings: list[LineString], curve_labels: list[bool],
    max_length_to_merge: float = 22,
    to_meters = True, utm_zone: int = None, input_crs: CRS = None,
    merge_lines = True,
    merge_curves = False,
) -> tuple[list[LineString], list[bool]]:
    """Given the output of split_by_curves functions, merge short segments making them part of
    the surrounding ones (and changing their label accordingly). By default, only merges short
    straight segments surrounded by curves.

    :param linestrings: List of line/curve segments, same length as curve_labels.
    :type linestrings: list[LineString]
    :param curve_labels: List of booleans representing if each segment is a curve or not.
    :type curve_labels: list[bool]
    :param max_length_to_merge: Max length for a segment to be considered short, defaults to 22
    :type max_length_to_merge: float, optional
    :param to_meters: Calculate distance in meters (will internally preprocess the linestrings to transform them into projected coordinates), defaults to True
    :type to_meters: bool, optional
    :param merge_lines: Merge straight segments, defaults to True
    :type merge_lines: bool, optional
    :param merge_curves: Merge curve segments, defaults to False
    :type merge_curves: bool, optional
    :return: The list of linestrings and booleans with short parts merged.
    :rtype: tuple[list[LineString], list[bool]]
    """
    if len(linestrings) != len(curve_labels):
        raise ValueError(f"length of linestrings different than curve_labels: {len(linestrings)} != {len(curve_labels)}")

    # Possibly more optimized than appending, as we already know max size
    out_linestrings = [None] * len(linestrings)
    out_labels = [None] * len(linestrings)

    # Preprocess to convert into meters
    if to_meters:
        if utm_zone is None:
            lines_center = np.array([line.centroid.coords[0] for line in linestrings]).mean(axis=0)
            meters_utm_zone = get_utm_zone(lines_center[0], lines_center[1])
        if input_crs is None:
            input_crs: CRS = CRS('EPSG:4326') # default to WGS84
        target_crs = CRS(proj='utm', zone=meters_utm_zone, units="m", ellipsis=input_crs.datum.ellipsoid)
        transformer = Transformer.from_crs(input_crs, target_crs, always_xy=True)
        linestrings_lengths = [transform(transformer.transform, line).length for line in linestrings]
    else:
        linestrings_lengths = [line.length for line in linestrings]

    # Generally speaking, loop through segments, 
    # and delay adding them to the output until they are
    # before the current index, until then they could be added
    # as the "previous segment" in merging

    i = 1
    j = 0
    while i < len(linestrings) - 1:
        prev_line, prev_label = linestrings[i - 1], curve_labels[i - 1]
        cur_line, cur_label, cur_length = linestrings[i], curve_labels[i], linestrings_lengths[i]
        can_merge = cur_label and merge_curves or not cur_label and merge_lines
        if can_merge and cur_length <= max_length_to_merge:
            next_line = linestrings[i + 1] # assumes prev_label = next_label, as segments are split up when different in original function
            merged_line = linemerge([prev_line, cur_line, next_line])
            out_linestrings[j] = merged_line
            out_labels[j] = prev_label

            j += 1
            i += 3 # skip to next possible center of merging
        else:
            # append prev line, as this could still be the "prev line" of the next value
            # if its short enough for merging
            out_linestrings[j] = prev_line
            out_labels[j] = prev_label
            j += 1
            i += 1
    
    # Last element (or two elements, depends) is the "previous" of nothing, so if it wasn't added as part of a merge add it
    while i-1 < len(linestrings):
        out_linestrings[j] = linestrings[i-1]
        out_labels[j] = curve_labels[i-1]
        i += 1
        j += 1

    out_linestrings = out_linestrings[:j]
    out_labels = out_labels[:j]

    return out_linestrings, out_labels


def _way_data_is_curve(way_data: dict | Series) -> bool:
    if ("junction" in way_data and way_data["junction"] in ["roundabout", "circular", "jughandle"]):
        return True
    return None

def get_point_in_route_gdf(split_route_gdf: GeoDataFrame, t: float|np.ndarray, geom_key="geometry", **getpointargs):
    if not ("iscurve" in split_route_gdf.columns):
        raise ValueError("split_route_gdf should be a GeoDataFrame with a iscurve column as returned by split_route_gdf_by_curves")
    
    # print("Using column", geom_key)
    # print(split_route_gdf[geom_key])
    # print(split_route_gdf[geom_key].describe())

    return get_point_in_linestrings_select_curves(split_route_gdf[geom_key].tolist(), split_route_gdf.iscurve.tolist(), t, **getpointargs)

def get_point_in_linestrings_select_curves(linestrings: list[LineString], curve_labels: list[bool], t: float|np.ndarray,
                                           continuous_lines=True,
                                           prev_linestring: LineString = None,
                                           next_linestring: LineString = None,
                                           return_label=False,
                                           return_linestring_indices=False,
                                           ) -> np.ndarray|tuple[np.ndarray,list[bool]]:
    """Return one or N points interpolated inbetween the various linestrings passed, treating them as curves
    if the corresponding label value is true and so interpolating them with bezier, or using line interpolation
    if the label is false.
    Optimized to run with more values, if you need to interpolate more points pass t as an ndarray instead of running
    this more times.

    Args:
        linestrings (list[LineString]): List of Linestrings.
        curve_labels (list[bool]): List of boolean values that are true if the Linestring at the same position
            should be a curve, must be as long as the linestrings list.
        t (float | np.ndarray): _description_
        continuous_lines (bool, optional): If the lines are continuous, make bezier curves match surrounding straight lines' vectors. Defaults to True.
        prev_linestring (LineString): Linestring to make the first segment be continuous with, if it is a curve. Defaults to None.
        next_linestring (LineString): Linestring to make the first segment be continuous with, if it is a curve. Defaults to None.
        return_label (bool, optional): Also return the label of each interpolated point as a separate list. Defaults to False.

    Returns:
        np.ndarray: interpolated point(s)
        list[bool]: if return_label is True, list of labels (curve or not) for each returned point
        list[int]: if return_linestring_indices is True, list of the index of the corresponding linestring for each returned point
    """
    
    if type(t) != np.ndarray and (t < 0 or t > 1):
        raise ValueError("t should be between 0 and 1.")
    elif type(t) == np.ndarray and np.any((t < 0) | (t > 1)):
        raise ValueError("t should be between 0 and 1.")
    if len(linestrings) != len(curve_labels):
        raise ValueError(f"linestrings should be the same length of curve_labels (were {len(linestrings)} and {len(curve_labels)} long)")
    
    precision = 1e-7
    
    coord_dims = len(linestrings[0].coords[0])
    total_len = sum([line.length for line in linestrings])
    t_len = t * total_len
    t_len_list = t_len.tolist() if type(t) == np.ndarray else [t_len]
    t_len_list.sort()
    out = np.empty((len(t_len_list), coord_dims))
    if return_label:
        out_labels = [False] * len(t_len_list)
    if return_linestring_indices:
        out_linestring_indices = [-1] * len(t_len_list)
        
    cur_dist = 0
    cur_idx = 0
    for (i, line), label in zip(enumerate(linestrings), curve_labels):
        # remove adjacent duplicates to make bezier library work (length stays the same)
        points_stripped = np.array(_remove_adjacent_duplicates(list(line.coords)))
        # add point "near" second-to-last point of prev line
        # (assuming last point matches this first point) and similary
        # near second point of next line
        # to better match direction between straight lines and bezier lines
        if continuous_lines:
            prev_line = linestrings[i-1] if i-1 >= 0 else prev_linestring
            next_line = linestrings[i+1] if (i+1 < len(linestrings)) else next_linestring
            bezier_prev = _close_point_towards(points_stripped[0], np.array(prev_line.coords[-2])) if prev_line else None
            bezier_next = _close_point_towards(points_stripped[-1], np.array(next_line.coords[1])) if next_line else None
        else:
            bezier_next, bezier_prev = None, None

        while t_len_list[cur_idx] - cur_dist <= line.length + precision:
            line_t = (t_len_list[cur_idx] - cur_dist) / line.length
            is_curve = label
            if is_curve:
                # Use bezier for 2d part, and linear for height (better results)

                # cap line_t to 1 to take precision margin into account
                point = _bezier_interpolate_check_3d(
                    points_stripped, min(line_t, 1),
                    bezier_prev, bezier_next,
                    coord_dims >= 3,
                    )
            else:
                point = linear_interpolation_polyline(points_stripped, min(line_t, 1))
            if return_label:
                out_labels[cur_idx] = is_curve
            if return_linestring_indices:
                out_linestring_indices[cur_idx] = i
            out[cur_idx] = point
            cur_idx += 1
            if cur_idx >= len(t_len_list):
                break
        if cur_idx >= len(t_len_list):
            break
        cur_dist += line.length
    
    if cur_idx < len(t_len_list):
        raise Exception(f"Interpolation: something went wrong, didn't finish interpolating all values"\
            f"(went up to {cur_idx}/{len(t_len_list)})")
    
    output = (np.array(out),)
    if return_label:
        output += (out_labels,)
    if return_linestring_indices:
        output += (out_linestring_indices,)

    if len(output) == 1:
        output = output[0]

    return output
    
def _bezier_interpolate_check_3d(points, t, prev_pt, next_pt, is_3d):
    """
    Interpolate 2d part with spline, 3d part with linear, as it 
    gives a better result (priority is matching street, and using 3d
    spline tended to sacrifice that).

    NOTE: This has the side effect of it technically returning non-matching
    2d positions and heights, as the parametric length percent of the curve
    is evaluated on xy and z separately; this doesn't in practice change much
    as the z position inbetween is normally interpolated anyways, if using this
    in the intended usecase.
    """
    if not is_3d:
        return interpolate_point_on_bezier(points, t, prev_pt, next_pt)
    else:
        points_xy = points[:, 0:2]
        prev_xy = prev_pt[0:2] if prev_pt is not None else None
        next_xy = next_pt[0:2] if next_pt is not None else None
        points_z = points[:, 2:3]

        if np.isscalar(t):
            return np.concatenate([
                interpolate_point_on_bezier(points_xy, t, prev_xy, next_xy),
                linear_interpolation_polyline(points_z, t)
            ])
        else:
            return np.concatenate([
                interpolate_point_on_bezier(points_xy, t, prev_xy, next_xy),
                linear_interpolation_polyline(points_z, t)
            ], axis=1)

def _remove_adjacent_duplicates(lst):
    i = 1
    while i < len(lst):
        if lst[i] == lst[i - 1]:
            del lst[i]
        else:
            i += 1
    return lst

def _close_point_towards(from_pt: np.ndarray, towards_pt: np.ndarray) -> np.ndarray:
    return _lerp(from_pt, towards_pt, 0.01)

def _lerp(x, y, t):
    return x * (1-t) + y * t

def linear_interpolation_polyline(line: shapely.LineString|np.ndarray, t: float|np.ndarray, by_length = True) -> np.ndarray:
    """Return one or more points (same dimension as t) in a linestring, at the given percentage
    of its length.

    Args:
        by_length (bool, optional): Treat t as percentage of length. If False, treats t as percentage alongside list indices. Defaults to True.
    """
    if type(t) != np.ndarray and (t < 0 or t > 1):
        raise ValueError("t should be between 0 and 1.")
    elif type(t) == np.ndarray and np.any((t < 0) | (t > 1)):
        raise ValueError("t should be between 0 and 1.")
    
    if type(line) is shapely.LineString:
        points = np.array(line.coords)  
    elif type(line) is np.ndarray:
        points = line
    else:
        points = np.array(line)
    
    n = len(points)
    if by_length:
        normalized_lengths = normalize_by_length(points)

        indices = np.interp(t, normalized_lengths, np.arange(n))
    else:
        r = np.arange(n)
        indices = np.interp(t, r / (n - 1), r)

    lower_indices = np.floor(indices).astype(int)
    upper_indices = np.ceil(indices).astype(int)
    frac = indices - lower_indices

    if np.isscalar(t):
        return (1 - frac) * points[lower_indices] + frac * points[upper_indices]
    else:
        return (1 - frac)[:, np.newaxis] * points[lower_indices] + frac[:, np.newaxis] * points[upper_indices]

def normalize_by_length(points):
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = np.sum(segment_lengths)
    if total_length > 0:
        # Add a leading zero to include all points
        return np.append(0, cumulative_lengths / total_length)
    else:
        return np.ones((len(points)))

def interpolate_point_on_bezier(line: shapely.LineString, t: (float|np.ndarray), 
                line_prev: shapely.LineString|np.ndarray=None,
                line_next: shapely.LineString|np.ndarray=None,
                neighbors_use_size = 1,
                spline_degree=3, transpose_result=True,
                ) -> np.ndarray:
    if type(t) != np.ndarray and (t < 0 or t > 1):
        raise ValueError("t should be between 0 and 1.")
    elif type(t) == np.ndarray and np.any((t < 0) | (t > 1)):
        raise ValueError("t should be between 0 and 1.")

    points_list = list(line.coords) if type(line) == shapely.LineString else line.tolist()
    line_len = _get_pt_list_length(points_list)
    # Continuous with prev/next street handling
    if line_prev is not None:
        isa = type(line_prev) is np.ndarray
        if isa and neighbors_use_size != 1:
            raise ValueError(f"neighbors_use_size must be 1 if line_prev is ndarray representing coords: is {neighbors_use_size}, line_prev is {line_prev}")
        points_list[0:0] = line_prev.coords[:-neighbors_use_size] if not isa else [tuple(line_prev)]
        # first N points + existing starting point
        added_len = _get_pt_list_length(points_list[:neighbors_use_size+1])
        t = (t * line_len + added_len) / (line_len + added_len)
        line_len += added_len
    if line_next is not None:
        isa = type(line_next) is np.ndarray
        if isa and neighbors_use_size != 1:
            raise ValueError(f"neighbors_use_size must be 1 if line_next is ndarray representing coords: is {neighbors_use_size}, line_next is {line_next}")
        if isa:
            points_list.append(tuple(line_next))
        else:
            points_list.extend(line_next.coords[:neighbors_use_size])
        added_len = _get_pt_list_length(points_list[-neighbors_use_size-1:])
        t = t * line_len / (line_len + added_len)
        line_len += added_len

    points = np.array(points_list)
    is_transposed = False

    if len(points) <= spline_degree:
        res, is_transposed = linear_interpolation_polyline(points, t), True
    else:
        res = _bezier_interp(points.transpose(), t, spline_degree, line)
    if transpose_result != is_transposed:
        res = res.transpose()
    
    return res

def _bezier_interp(transposed_points: np.ndarray, t, k, line) -> np.ndarray:
    try:
        tck, _ = scint.splprep(transposed_points, s=0, k=k)
    except Exception as e:
        print("Input was ", transposed_points, line)
        raise e
    return np.array(scint.splev(t, tck))

def _get_pt_list_length(lst: list[tuple[float, float]]):
    list_arr = np.array(lst)
    return sum(np.linalg.norm(list_arr[1:] - list_arr[:-1], axis=1))

def add_elevation_to_gdf_geometry(gdf: GeoDataFrame, 
                                  geom_key = "geometry", start_ele_key = "u_elevation", end_ele_key = "v_elevation", 
                                  keep_old_geom_key = "geometry2d",
                                  ):
    """Add elevation to the Linestrings in a gdf that contains columns for elevation at the start and end of each row.
    Interpolates between values in the linestring.
    """

    if keep_old_geom_key:
        gdf[keep_old_geom_key] = gdf[geom_key]

    gdf[geom_key] = gdf.apply(lambda row: _get_linestring_with_height(row[geom_key], row[start_ele_key], row[end_ele_key]), axis=1)

    return gdf


def _get_linestring_with_height(linestring2d: LineString, start_height: float, end_height: float):
    points = np.array(linestring2d.coords)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = linestring2d.length
    pcts = np.concatenate([[0], cumulative_lengths / total_length])
    n = len(points)

    return LineString([(points[i, 0], points[i, 1], _lerp(start_height, end_height, pcts[i])) for i in range(n)])