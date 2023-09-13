import pandas as pd
import itertools
import warnings
import functools
from shapely import LineString, Point
import itertools
import numpy as np

debug = False

# Utilities
def first(iterable):
    for x in iterable:
        return x

def logdebug(*args, **pargs):
    global debug
    if debug:
        print(*args, **pargs)

def lerp(x, y, t):
    return x * (1 - t) + y * t

def lerp_scale(x, y, t, left=0, right=1):
    t = max(0, min(1, (t - left) / (right - left)))
    return x * (1 - t) + y * t

# test_list = [(4, 2, 23423), (25, 4, 12423432), (2, 11, 64563), (11, 17, 12423)]
# test_list2_1 = [4, 25, 2, 11]
# test_list2_2 = [2, 4, 11, 17]

def extract_connections_sequence(points_u: list[int]|pd.Series, points_v: list[int]|pd.Series, try_connect_broken=True) -> list[int]:
    """Extract a sequence from lists of numbers such that each point in u
    is equal to another in v (except one in each)
    """

    if type(points_u) == pd.Series:
        points_u = points_u.to_list()
    if type(points_v) == pd.Series:
        points_v = points_v.to_list()

    # Mapping of first numbers to pointers
    # num_dict = {u: i for i, u in enumerate(points_u)} # Simple version where no more than one edge has same u
    # Complicated version: to handle cases like this with loops https://www.openstreetmap.org/way/302328448#map=19/44.52739/11.29592
    num_dict: dict[int, list[int]] = {num: [i for i, x in enumerate(points_u) if x == num] for num in points_u}
    visited = {}

    # try_connect_broken: list of building parts, else: single building segment
    out = []

    # Find the index with unique first number
    for i, u in enumerate(points_u):
        if u not in points_v:
            if try_connect_broken:
                out.append([u, points_v[i]])
            else:
                out.extend([u, points_v[i]])
                break

    # No unique first number: loop, random start
    if len(out) == 0:
        # fixed arbitrary number for repeatability
        r = int(0.33 * len(points_u))
        if try_connect_broken:
            out.append([points_u[r], points_v[r]])
        else:
            out.extend([points_u[r], points_v[r]])

    # Reconstruct the sequence
    if try_connect_broken:
        for seg in out:
            next_num = seg[-1]
            while True:
                next_i = next(filter(lambda x: x not in visited, num_dict.get(next_num, [])), None)
                if next_i is not None:
                    seg.append(points_v[next_i])
                    visited[next_i] = True
                    next_num = points_v[next_i]    
                else:
                    break  # Sequence cannot be completed
        if len(out) == 1:
            return out[0]
        else:
            _sort_by_sequence_closeness(out)
            return [item for sublist in out for item in sublist]

    else:
        next_num = out[-1]
        while True:
            next_i = next(filter(lambda x: x not in visited, num_dict.get(next_num, [])), None)
            if next_i is not None:
                out.append(points_v[next_i])
                visited[next_i] = True
                next_num = points_v[next_i]    
            else:
                break  # Sequence cannot be completed

        return out

# currently not too efficient, but not meant for mass use out of data analisys for now
def _sort_by_sequence_closeness(lst: list[list[int]], maxiters=None):
    if not maxiters:
        maxiters = len(lst)
    if len(lst) == 2:
        lst[:], _ = _swap_in_order(*lst)
    else:
        for i in range(maxiters):
            did_swap = False
            for j in range(len(lst) - 1):
                lst[j:j+2], swapped = _swap_in_order(lst[j], lst[j + 1])
                did_swap = did_swap or swapped
            if not did_swap:
                break

def _swap_in_order(la: list[int], lb: list[int]):
    return ((lb, la), True) if abs(la[0] - lb[-1]) < abs(la[-1] - lb[0]) \
        else ((la, lb), False)

def sorted_list_of_connections(lst: list[tuple]):
    num_dict = {tup[0]: tup for tup in lst}  # Mapping of first numbers to tuples
    result = []

    # Find the tuple with the unique first number
    secnums = [x[1] for x in lst]
    for tup in lst:
        if tup[0] not in secnums:
            result.append(tup)
            break

    # Reconstruct the sequence
    while len(result) < len(lst):
        next_num = result[-1][1]
        next_tup = num_dict.get(next_num)
        if next_tup:
            result.append(next_tup)
        else:
            break  # Sequence cannot be completed

    return result

def get_segment_indices(list_of_lists: list[list]|list[LineString], overlap: bool = None) -> list[tuple[int, int]]:
    """From a list of lists representing a singular list that was split in parts, return the indices
    you'd need to use in slicing to obtain that part from the original list:
    ie
    ```python
    bounds = get_segment_indices(split_list)[3]
    print(original_list[bounds[0]:bounds[1]] == split_list[3]) #true
    ```
    Can also work with lists of shapely Linestrings.

    Arguments:
        list_of_lists: input, see above.
        overlap: if each part's last element is the same as the next one; will ignore one element for each
            inbetween part if so. If not set, defaults to True if list of linestrings, False otherwise.
    """
    is_linestring = type(list_of_lists[0]) is LineString
    _len = (lambda x: len(x.coords)) if is_linestring else len
    if overlap is None:
        overlap = is_linestring

    if overlap:
        starts = [0]
        starts.extend(starts[-1] + _len(sublist) - 1 for sublist in list_of_lists[:-1])
        # starts = itertools.islice(itertools.chain([0], itertools.accumulate(map(_len, list_of_lists[:-1]))), len(list_of_lists))
    else:
        starts = [0] + [sum(_len(sublist) for sublist in list_of_lists[:i]) for i in range(1, len(list_of_lists))]

    indices = [(start, start + _len(sublist)) for start, sublist in zip(starts, list_of_lists)]
    return indices

def curvature(line: np.ndarray):
    """Get the curvature (approximated, as we are using discrete points instead of a continuous formula)
    of points in a line. Formulas for reference are:
    - curvature: :math:`k := \frac{1}{R}`
    - radius of curvature (parametrical*): :math:`R = \left | \frac{(\dot{x}^2 + \dot{y}^2)^{3/2}}{\dot{x}\ddot{y} - \dot{y}\ddot{x}} \right |`

    Example reference: [wikipedia](https://en.wikipedia.org/wiki/Radius_of_curvature)

    *parametrical here because list of points, the parameter being the index, more or less.

    :param line: Array representing coordinates of point in line, should have shape like this: [[x1, y1], [x2, y2], ...]
    :type line: np.ndarray
    """

    # axis=0 same as doing it separately for each array of coordinates (every x, every y, ec)
    first_der = np.gradient(line, axis=0)
    second_der = np.gradient(first_der, axis=0)

    xd, yd = (first_der[:, 0], first_der[:, 1])
    xdd, ydd = (second_der[:, 0], second_der[:, 1])

    return np.abs((xd * ydd - yd * xdd) / (xd * xd + yd * yd)**1.5 )

def get_utm_zone(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return zone

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func