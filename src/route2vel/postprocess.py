from geopandas import GeoDataFrame
from shapely import Point
from shapely.ops import transform
import shapely
import numpy as np
from .utils import curvature
from collections.abc import Callable

def clip_interp_gdf(gdf: GeoDataFrame, bounds: tuple[tuple[float, float], tuple[float, float]], transform_bounds:Callable[[float, float], tuple[float, float]]=None) -> GeoDataFrame:
    bounds = shapely.Polygon([bounds[0], (bounds[1][0], bounds[0][1]), bounds[1], (bounds[0][0], bounds[1][1])])
    if transform_bounds is not None:
        bounds = transform(transform_bounds, bounds)
    crop: GeoDataFrame = gdf.clip(bounds, True).sort_index()
    return crop

def offset_meters_gdf(gdf: GeoDataFrame) -> GeoDataFrame:
    x_min = min([pt.x for pt in gdf.geometry])
    y_min = min([pt.y for pt in gdf.geometry])

    return gdf.assign(
        geometry = [Point(pt.x - x_min, pt.y - y_min, pt.z) for pt in gdf.geometry],
    )

def calc_curvature(gdf: GeoDataFrame, on_place=True) -> None|GeoDataFrame:
    line = np.array([(pt.x, pt.y) for pt in gdf.geometry])
    curv = curvature(line)
    if on_place:
        gdf['curvature'] = curv
    else:
        return gdf.assign(curvature=curv)

def interp_gdf_to_csv(
    gdf: GeoDataFrame, fname: str, 
    separate_roundabout: bool = False,
    add_tract_start_col: bool = False,
    extra_cols: list[str] = [],
):
    with open(fname, 'w', encoding='utf-8') as f:
        cols = ['lon','lat','ele','iscurve']
        if add_tract_start_col:
            cols.append('start_line')
        cols += extra_cols
        print(','.join(cols), file=f)

        print(f"Columns: {cols}")

        tract_start = -1
        last_is_curve = None
        line_num = 0
        for index, row in gdf.iterrows():
            line_num += 1
            point: Point = row['geometry']
            is_curve = row['is_curve']
            curve_category = 'curve' if is_curve else 'line'
            if separate_roundabout and row['junction'] == 'roundabout':
                curve_category = 'roundabout'

            if last_is_curve != curve_category:
                last_is_curve = curve_category
                tract_start = line_num

            data = [
                point.x, point.y, point.z,
                curve_category,
            ]
            if add_tract_start_col:
                data.append(tract_start)
            for col in extra_cols:
                data.append(row[col])
            
            print(','.join(map(str, data)), file=f)
            
    print(f"Written {fname}")