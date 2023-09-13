import matplotlib
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
import geopandas as gpd
import shapely
import contextily as cx
import numpy as np
import scipy

def x_to_colors(X, cmap):
    return [matplotlib.colormaps.get_cmap(cmap)(x / len(X)) for x in range(len(X))]

def make_better_barh(ax, X, Y, title: str = None, cmap: str = None, pct = False, **kwargs):
    if cmap:
        colors = x_to_colors(X, cmap)
        kwargs['color'] = colors

    bar = ax.barh(X, Y, **kwargs)
    
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    ax.grid(color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    
    ax.invert_yaxis()
    
    for i in ax.patches:
        ax.annotate(str(f'{round((i.get_width()), 2)}%' if pct else str(round(i.get_width(), 2))),
                (i.get_width()+0.75, i.get_y()+i.get_height() / 2),
                va='center',
                fontsize = 10, fontweight ='bold',
                color ='lightgray')
        
    if title:
        ax.set_title(title)

    if pct:
        ax.set_xlim([0, 100])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    return bar

# zoomin
# gdf: gdf or list of gdf,kwargs (for plot)
def display_bounded_shapes(gdf: gpd.GeoDataFrame|list[(gpd.GeoDataFrame,dict)], boundsTl, boundsBr, crs=None, latLong=False, scale=1, bgalpha=1, figsize=(10,10)) -> gpd.GeoDataFrame:
    if latLong:
        boundsTl = boundsTl[::-1]
        boundsBr = boundsBr[::-1]
    shape = shapely.Polygon([boundsTl, (boundsBr[0], boundsTl[1]), boundsBr, (boundsTl[0], boundsBr[1])])
    shape = shapely.affinity.scale(shape, xfact=scale, yfact=scale)

    if type(gdf) != list:
        gdf = [(gdf, {})]

    if crs is None:
        crs = gdf[0][0].crs

    fig, ax = plt.subplots(figsize=figsize)
    cropped_gdfs = []
    for t in gdf:
        gdf_i = t[0]
        args: dict = t[1]
        label_params=None
        if len(t) > 2:
            label_params = t[2]
        gdf_cropped = gdf_i.clip(shape)
        if len(gdf_cropped) == 0:
            print("gdf isn't included in bounds!")
            continue
        if 'edgecolor' not in args and 'color' not in args and 'column' not in args:
            args['edgecolor'] = 'teal'
        gdf_cropped.plot(ax=ax, **args)
        
        if label_params and label_params['column'] in gdf_cropped:
            alt = True
            for idx, row in gdf_cropped.iterrows():
                alt = not alt
                offset_x, offset_y = label_params.get('offset', (0,0))
                if 'altOffset' in label_params and alt:
                    offset_x, offset_y = label_params['altOffset']
                x, y = row.geometry.centroid.x, row.geometry.centroid.y
                fontsize = label_params.get('size', 10)
                label = row[label_params['column']]
                ax.text(
                    x + offset_x, y + offset_y, str(label), 
                    fontsize=fontsize, 
                    color='black', ha='center', va='center'
                )
            
        cropped_gdfs.append(gdf_cropped)
    
    gpd.GeoSeries(shape, crs=crs).plot(ax=ax, facecolor='none', edgecolor='r')
    cx.add_basemap(ax=ax, crs=crs, source=cx.providers.OpenStreetMap.Mapnik, alpha=bgalpha)
    plt.show()

    return cropped_gdfs



def plot_interpolate_bezier(gdf, points_per_edge = 100, degree = 3, figsize=(12,12), geomkey='geometry', show=True):
    points = []
    colors = []
    for geom in gdf[geomkey]:
        xs, ys = interpolate_point_on_bezier(geom, np.linspace(0, 1, points_per_edge), transpose_result=False, spline_degree=degree)
        points.append((xs, ys))
        if len(geom.coords) <= degree:
            colors.append('r')
        else:
            colors.append('g')
    print(f"Interpolated {sum(len(edge[0]) for edge in points)} points in {len(gdf[geomkey])} edges")

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, color='k')
    for edge, color in zip(points, colors):
        ax.plot(*edge, color=color)
    cx.add_basemap(ax=ax, crs=gdf.crs, alpha=0.75, source=cx.providers.OpenStreetMap.Mapnik)
    if show:
        plt.show()

def plot_interpolate_geom_bezier(linestring: shapely.LineString, crs, num_points = 1000, degree = 3, figsize=(12,12), show=True):
    points = []
    colors = []
    xs, ys = interpolate_point_on_bezier(linestring, np.linspace(0, 1, num_points), transpose_result=False, spline_degree=degree)
    points.append((xs, ys))
    if len(linestring.coords) <= degree:
        colors.append('r')
    else:
        colors.append('g')

    fig, ax = plt.subplots(figsize=figsize)
    gpd.GeoDataFrame({'geometry': [linestring]}).plot(ax=ax, color='k')
    for edge, color in zip(points, colors):
        ax.plot(*edge, color=color)
    cx.add_basemap(ax=ax, crs=crs, alpha=0.75, source=cx.providers.OpenStreetMap.Mapnik)
    if show:
        plt.show()

def interpolate_point_on_bezier(line: shapely.LineString, t: (float|np.ndarray), 
                line_prev: shapely.LineString=None,
                line_next: shapely.LineString=None,
                neighbors_use_size = 1,
                spline_degree=3, transpose_result=True
                ):
    if type(t) != np.ndarray and (t < 0 or t > 1):
        raise ValueError("t should be between 0 and 1.")
    elif type(t) == np.ndarray and np.any((t < 0) | (t > 1)):
        raise ValueError("t should be between 0 and 1.")

    points_list = list(line.coords) if type(line) == shapely.LineString else line
    pstart, pend = 0, len(points_list)
    # Continuous with prev/next street handling
    if line_prev:
        points_list[0:0] = line_prev.coords[:-neighbors_use_size]
        pstart += neighbors_use_size
        pend += neighbors_use_size
        t = (t - 1) / (1 - neighbors_use_size)
    if line_next:
        points_list.extend(line_next.coords[:neighbors_use_size])
        t = t / (1 + neighbors_use_size)

    points = np.array(points_list)
    transposed = points.transpose()

    if len(points) <= spline_degree:
        res = linear_interpolation_polyline(points, t).transpose()
    else:
        tck, u = scipy.interpolate.splprep(transposed, s=0, k=spline_degree)
        res = scipy.interpolate.splev(t, tck)
    if transpose_result:
        return res.transpose()
    else:
        return res
    

def linear_interpolation_polyline(line: shapely.LineString|np.ndarray, t: float|np.ndarray):
    points = list(line.coords) if type(line) == shapely.LineString else line
    n = len(points)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative_lengths = np.cumsum(segment_lengths)
    total_length = np.sum(segment_lengths)
    # Add a leading zero to include all points
    normalized_lengths = np.append(0, cumulative_lengths / total_length)

    indices = np.interp(t, normalized_lengths, np.arange(n))
    lower_indices = np.floor(indices).astype(int)
    upper_indices = np.ceil(indices).astype(int)
    frac = indices - lower_indices

    return (1 - frac)[:, np.newaxis] * points[lower_indices] + frac[:, np.newaxis] * points[upper_indices]