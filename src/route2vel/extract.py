"""
Deprecated: functionality replaced by interp and others. Keeping this around as logic can
be reused in developing other modules.
"""
import warnings
warnings.warn("the extract module is deprecated", DeprecationWarning, stacklevel=2)

from shapely.geometry import LineString
import numpy as np
import heapq
import networkx as nx
import math

from .route import measure_route, RouteEdge, RouteNode, RouteSegment
from .utils import logdebug, lerp_scale

def route_to_pos_spd(
        route: list[RouteSegment],
        use_height = False,
        dist_check = False, 
        ) -> list[dict]:
    """Convert a dict list route (for format see `route.route_to_edges`) to a list of 
    positions and maxspeeds at each position. The positions are sampled at each node.
    Position is 3d, height will simply be 0 if not used.

    Args:
        route (list[RouteSegment]): list returned by `route.route_to_edges`.
        use_height (bool, optional): Use height data. Defaults to False.
        dist_check (bool, optional): If distance in geometry and 
            edge length should be checked to be matching. Defaults to False.

    Returns:
        list[dict]: List of dicts with this format: {
            pos: np.array (long, lat, height) (osmnx uses lat as y and long as x),
            speed: int, speed limit in km/h (osm units)
            dist: float (from starting point, in km, used for velocity falloff)
            node_id: int? Node id of the osmnx graph, if any (not present in geometry positions)
        }
    """
    out = []

    curdist = 0

    for segment in route:
        fromnode = segment.from_node
        tonode = segment.to_node # for height interpolation
        # Assume geometry is present, error otherwise
        # (To avoid errors; use fill from loading if necessary)
        geom = segment.edge.geometry
        geom_len_coord = geom.length
        edge_len_m = segment.edge.length
        # Shared by all positions in this iteration
        speed = int(segment.edge.speed_kph)

        # For height: cannot just default to 0 if height_key is not present without using use_height,
        # because in that case some nodes might end up at height 0 while others at a "normal" height,
        # leading to wildly different positions
        from_height = fromnode.elevation if use_height else 0
        to_height = tonode.elevation if use_height else 0

        # separately handle start point to insert node id
        out.append({
            "pos": np.array(fromnode.coords3d),
            "speed": speed,
            "dist": curdist,
            "node_id": fromnode.id,
        })

        lastpos = np.array(fromnode.coords)

        # do not add first and last, usually they are the nodes
        for i, pos in enumerate(geom.coords[1:-1]):
            # Calc curdist increase depending on coordinate shift from last pos
            # do not consider height as it's interpolated too
            curpos = np.array(pos[:2])
            prog = np.linalg.norm(curpos - lastpos) / geom_len_coord
            curdist += prog * edge_len_m 
            out.append({
                # also interpolate height using prog
                "pos": np.array((pos[0], pos[1], from_height * (1 - prog) + to_height * prog)),
                "dist": curdist,
                "speed": speed,
            })
            lastpos = curpos

        # Increase distance at last pos
        endpos = np.array(tonode.coords)
        curdist += np.linalg.norm(endpos - lastpos) * edge_len_m / geom_len_coord
        
    if dist_check:
        route_dist = measure_route(route, "length")
        if abs(route_dist - curdist) > 1e-6: # account for approx error
            raise Exception(f"Failed distance check: should be {route_dist}, is {curdist}")

    # Append last node
    last_node = route[-1].to_node
    last_node_id = last_node.id
    last_speed = route[-1].edge.speed_kph
    out.append({
        "pos": np.array(last_node.coords3d),
        "dist": curdist,
        "speed": last_speed,
        "node_id": last_node_id,
    })
    return out

def _get_extra_samples(cur: dict, next: dict, sample_dist: float):
    to_add = math.ceil((next['dist'] - cur['dist']) / sample_dist)
    for i in range(to_add):
        sample = {
            'dist': cur['dist'] + sample_dist * i,
            'speed': cur['speed'],
            'pos': lerp_scale(cur['pos'], next['pos'], i, 0, to_add),
        }
        # if i == 0:
        if i == 0 and 'node_id' in cur:
            sample['node_id'] = cur['node_id']
        yield sample

def add_pos_vel_samples(
        pos_spd: list[dict],
        sample_dist: float = 8,
        sample_dist_nth_smallest: int = 20,
        ) -> list[dict]:
    """Add more samples to a list returned by `route_to_pos_spd`, to avoid too far samples
    as for example generated by `loading.fill_geometry`.


    Args:
        pos_spd (list[dict]): List as returned by `route_to_pos_spd`.
        sample_dist (float, optional):  Distance between extra samples (in meters). If None, 
            will use `sample_dist_nth_smallest` smallest dist in route geometry. Defaults to 8.
        sample_dist_nth_smallest (int, optional): See `sample_dist`. Defaults to 20.
        tolerance_dist (float, optional): dist tolerance for adding samples. Defaults to 1e-2.

    Returns:
        list[dict]: list with added samples
    """
    if sample_dist is None:
        # get third smallest dist between nodes
        sample_dist = heapq.nsmallest(sample_dist_nth_smallest, map(lambda x: x[1]['dist'] - x[0]['dist'], zip(pos_spd, pos_spd[1:])))[-1]
        logdebug(f"Sample dist set to {sample_dist:.2f}m ({sample_dist_nth_smallest}-th smallest)")
    else:
        logdebug(f"Sample dist is {sample_dist:.2f}m")

    out = [sample for cur, next in zip(pos_spd, pos_spd[1:]) for sample in _get_extra_samples(cur, next, sample_dist)]
        
    out.append(pos_spd[-1])

    return out

KM_H_2_M_S = 1 / 3.6

preset_driving_settings = {
    'default': {
        'stop_distance': 150, #meters, distance where vehicle starts slowing down before stop
        'slow_distance': 80, #meters, distance where vehicle starts slowing down before/after lower speed limit is reached
        'accelerate_distance': 120, #meters, distance where vehicle starts accelerating before/after higher speed limit is reached
        'slow_after_change': True, # If slow down starts after or before max speed is lowered
        'accelerate_after_change': True, # If speed up starts after or before max speed is increased
        'speed_limit_treshold': 5, # offset above/below the speed limit where the vechile tries to stay
    },
}

def pos_spd_to_vel(
        pos_spd: list[dict],
        driving_settings: str|dict = 'default',
        stop_points: list[dict|int] = [-1],
        to_meters_second = False,
        return_metadata = False,
        ) -> list[np.ndarray]:
    """Convert list of position/maxspeed data (from `extract.route_to_pos_spd`) into a list of velocities 
    at each sample point

    Args:
        pos_spd (list[dict]): list as returned by route_to_pos_spd
        driving_settings (str | dict, optional): Name of the driving settings preset in `extract.preset_driving_settings.` 
            Defaults to 'default'.
        stop_points (list[dict | int], optional): a list of either pos_vel indices as ints or 
            dicts representing an osm id {'osmid': int}, represents points where the vehicle 
            stops, or is likely to (traffic lights, etc). Defaults to [-1] (only last point).
        to_meters_second (bool, optional): return output in m/s instead of km/h. Defaults to False.
        return_metadata (bool, optional): Return additional metadata. Defaults to False.

    Returns:
        list[np.ndarray]: list of velocity np.arrays (long, lat, height) in km/h or m/s (assuming maxspeed in osm was in km/h)
        list[dict]?: if return_metadata is True, additional info about the output
    """

    out = []
    metadata = []

    _driving_settings = driving_settings if type(driving_settings) == dict else preset_driving_settings[driving_settings]

    stop_indices = {}
    stop_osmids = {}

    for pt in stop_points:
        if type(pt) == int:
            if pt < 0:
                pt = len(pos_spd) + pt
            stop_indices[pt] = True
        else:
            stop_osmids[pt['osmid']] = True

    speed = 0 #in km/h, regardless of parameters
    speed_target = pos_spd[0]['speed'] + _driving_settings['speed_limit_treshold']
    speed_step = speed_target / _driving_settings['accelerate_distance'] # speed increase per meter of distance progressed
    next_change_idx = _get_next_speed_change(pos_spd, 0, stop_indices, stop_osmids)
    last_limit = pos_spd[0]['speed']
    started_following_next = False # updated where X_after_change is false, so used when slowing/etc before speed limit instead of after

    prev_dist = 0

    for (i, cur), next in zip(enumerate(pos_spd), pos_spd[1:]):
        diff = next['pos'] - cur['pos']
        norm = np.linalg.norm(diff)

        dist_increase = cur['dist'] - prev_dist
        dist_to_next_change = None

        resume_after_stop = speed_target == 0

        if next_change_idx:
            # also count dist_increase as this effects the speed change that happens in the same iteration
            dist_to_next_change = pos_spd[next_change_idx]['dist'] - cur['dist'] + dist_increase
            if (not started_following_next):
                new_speed_target = None
                if (_is_stop_point(next_change_idx, pos_spd, stop_indices, stop_osmids)
                    and dist_to_next_change <= _driving_settings['stop_distance']
                    ):
                    new_speed_target = 0
                elif (not _driving_settings['accelerate_after_change'] and dist_to_next_change <= _driving_settings['accelerate_distance']
                    and pos_spd[next_change_idx]['speed'] > cur['speed']
                    ):
                    new_speed_target = pos_spd[next_change_idx]['speed'] + _driving_settings['speed_limit_treshold']
                elif (not _driving_settings['slow_after_change'] and dist_to_next_change <= _driving_settings['slow_distance']
                    and pos_spd[next_change_idx]['speed'] < cur['speed']
                    ):
                    new_speed_target = pos_spd[next_change_idx]['speed'] + _driving_settings['speed_limit_treshold']

                if new_speed_target is not None:
                    speed_target = new_speed_target
                    speed_step = abs(speed_target - speed) / dist_to_next_change
                    started_following_next = True
                elif resume_after_stop:
                    speed_target = cur['speed'] + _driving_settings['speed_limit_treshold']
                    speed_step = abs(speed_target - speed) / _driving_settings['accelerate_distance']


            # Reached node where speed limit changes
            if i == next_change_idx:
                next_change_idx = _get_next_speed_change(pos_spd, i, stop_indices, stop_osmids)
                # if stop point, wait to affect speed as it would change it this iteration
                # (where the vehicle is supposed to stop)
                if not _is_stop_point(i, pos_spd, stop_indices, stop_osmids):
                    if _driving_settings['slow_after_change'] and cur['speed'] < last_limit:
                        speed_target = cur['speed'] + _driving_settings['speed_limit_treshold']
                        speed_step = abs(speed_target - speed) / _driving_settings['slow_distance']
                    elif _driving_settings['accelerate_after_change'] and cur['speed'] > last_limit:
                        speed_target = cur['speed'] + _driving_settings['speed_limit_treshold']
                        speed_step = abs(speed_target - speed) / _driving_settings['accelerate_distance']
                last_limit = cur['speed']
                started_following_next = False
      
        if speed_target > speed:
            speed = min(speed + speed_step * dist_increase, speed_target)
        elif speed_target < speed:
            speed = max(speed - speed_step * dist_increase, speed_target)
        # no need to reset speed step after reaching target, ifs will simply not pass

        vel = diff * speed / norm if norm > 0 else np.zeros((3,))
        if to_meters_second: 
            vel = vel * KM_H_2_M_S
        out.append(vel)
        if return_metadata:
            metadata.append({
                'speed_target': speed_target,
                'speed_step': speed_step,
                'speed': speed,
                'speed_limit': cur['speed'],
                'distance': cur['dist'],
                'dist_increase': dist_increase,
                'next_change_idx': next_change_idx,
                'next_speed': pos_spd[next_change_idx]['speed'] if next_change_idx else None,
                'next_stop': _is_stop_point(next_change_idx, pos_spd, stop_indices, stop_osmids) if next_change_idx else None,
                'dist_to_next_change': dist_to_next_change,
                'node_id': cur['node_id'] if 'node_id' in cur else None,
                'stop': _is_stop_point(i, pos_spd, stop_indices, stop_osmids),
            })

        prev_dist = cur['dist']

    out.append(np.zeros((3,)))
    if return_metadata:
        metadata.append({
            'speed_target': 0,
            'speed_step': None,
            'speed': 0,
            'speed_limit': cur['speed'],
            'distance': pos_spd[-1]['dist'],
            'dist_increase': pos_spd[-1]['dist'] - prev_dist,
            'next_change_idx': None,
            'next_speed': None,
            'dist_to_next_change': None,
            'node_id': pos_spd[-1]['node_id'] if 'node_id' in pos_spd[-1] else None,
            'stop': _is_stop_point(len(pos_spd) - 1, pos_spd, stop_indices, stop_osmids),
        })

    if return_metadata:
        return out, metadata
    return out

def _get_next_speed_change(pos_spd: list[dict], from_idx: int, stop_indices: dict[int, bool], stop_osmids: dict[int, bool]):
    for i in range(len(pos_spd) - from_idx - 2):
        idx = i + 2 + from_idx
        val = pos_spd[idx]
        prev = pos_spd[idx - 1]
        if (
            prev['speed'] != val['speed'] 
            or _is_stop_point(idx, pos_spd, stop_indices, stop_osmids)
        ):
            return idx
    return None
        
def _is_stop_point(idx: int, pos_spd: list[dict], stop_indices: dict[int, bool], stop_osmids: dict[int, bool]):
    return idx in stop_indices or ('node_id' in pos_spd[idx] and pos_spd[idx]['node_id'] in stop_osmids)

def get_stop_points(
    pos_spd: list[dict], route: list[RouteSegment], graph: nx.MultiDiGraph,
    traffic_lights: bool = True,
    stop_at_end: bool = True
    ) -> list[int|dict]:
    """Return stop points in a route to be used in pos_spd_to_vel, including
    things like traffic lights, probabilistic random stops, etc.

    Args:
        pos_spd (list[dict]): list as returned by route_to_pos_spd
        route (list[RouteSegment]): list returned by `route.route_to_edges`.
        graph (nx.MultiDiGraph): Osmnx graph.
        traffic_lights (bool, optional): add stop points at traffic lights. Defaults to True.
        stop_at_end (bool, optional): add stop point at the end of the route. Defaults to True.
        
        Returns []
    """
    
    out = []
    
    for i, el in enumerate(pos_spd):
        if 'node_id' in el:
            node_data = graph.nodes[el['node_id']]
            # info about *traffic_signals*: https://wiki.openstreetmap.org/wiki/Tag:highway%3Dtraffic_signals
            if traffic_lights and node_data.get('highway', None) == 'traffic_signals':
                out.append(i)
                
    if stop_at_end:
        out.append(-1)
                
    return out