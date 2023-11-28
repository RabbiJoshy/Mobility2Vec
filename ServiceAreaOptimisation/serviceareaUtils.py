import matplotlib.pyplot as plt
import requests
from shapely.geometry import Polygon, Point, box
import pyproj
from shapely import wkt
import json
from shapely.ops import unary_union

def plotjourneys(df, background, ServiceArea, modechoice = 'Personenauto - bestuurder', samples = 0,
                 choicecol = 'khvm', time = False, transit = False):
    best = df[df[choicecol] == modechoice]
    # best = best.set_index('vertpc').join(best.groupby('vertpc').count().iloc[:, :1].rename(columns={'walk_distance': 'vertcount'}))
    background = background.join(
        best.groupby('aankpc').count().iloc[:, :1].rename(columns={'walk_distance': 'aankcount'}))
    fig, ax = plt.subplots()
    DIEMEN.plot(ax = ax, facecolor = 'y', alpha = 0.1, linewidth = 0.05)
    if transit == True:
        tl['Marker_Size'] = tl['Modaliteit'].map({'Tram': 0.25, 'Metro': 5})
        tl['Color'] = tl['Modaliteit'].map({'Tram': 'y', 'Metro': 'y'})
        tl.plot(ax=ax, c=tl['Color'], markersize=tl['Marker_Size'], legend=True)
    background.plot(ax = ax, linewidth = 0.1, column = 'aankcount', alpha = 0.1) #facecolor = 'None'
    ServiceArea.plot(ax=ax, linewidth=0.1, alpha = 0.5)
    # ax.set_aspect('equal')
    if samples > 0:
        best = best.sample(min(len(best), samples))
    fracdict = countfrac(best, AP, ServiceArea)
    asize = best.groupby('aankpc').count().iloc[:, 0]
    vsize = best.groupby('vertpc').count().iloc[:, 0]
    x2 = AP.loc[best['aankpc']].geometry.centroid.x
    y2 = AP.loc[best['aankpc']].geometry.centroid.y
    x1 = AP.loc[best['vertpc']].geometry.centroid.x
    y1 = AP.loc[best['vertpc']].geometry.centroid.y
    ax.scatter(x1, y1, c = 'r', s = vsize[best['vertpc']])
    ax.scatter(x2, y2, c='g', s=asize[best['aankpc']])
    ax.set_title(str(fracdict))

    return

def mapplot(background, foreground):
    fig, ax = plt.subplots()
    background.plot(ax = ax, facecolor = 'None')
    foreground.plot(ax = ax)

def simplify_geometry(geom, tolerance=0.1):
    simplified = geom.simplify(tolerance, preserve_topology=True)
    simplified = simplified.buffer(0)
    # If the polygon is a MultiPolygon, we unify it into a single Polygon
    if simplified.geom_type == 'MultiPolygon':
        largest_polygon = None
        max_area = 0

        for polygon in simplified.geoms:
            if polygon.area > max_area:
                largest_polygon = polygon
                max_area = polygon.area

        simplified = largest_polygon
        # simplified = unary_union(simplified)

    return simplified

def polygon_to_overpass_format(polygon):
    # Extract exterior coordinates from polygon and reorder them to (lat, lon)
    exterior_coords = [(y, x) for x, y in list(polygon.exterior.coords)]

    # Convert coordinates to text and concatenate with Overpass polygon format
    overpass_polygon_format = "{}".format(" ".join([f"{lat} {lon}" for lat, lon in exterior_coords]))

    return overpass_polygon_format

def overpass_polygon(polygon):
    overpass_polygon = polygon_to_overpass_format(polygon)
    # print(overpass_polygon)

    overpass_query = f'''
    [out:json];
    (
      node(poly:"{overpass_polygon}");
    );
    out tags;
    '''

    overpass_url = "http://overpass-api.de/api/interpreter"
    response = requests.get(overpass_url, params={'data': overpass_query})
    # print(response)
    data = response.json()

    return data

def count_tags(data, t = "amenity"):
    amenity_counts = {}
    for element in data["elements"]:
        if "tags" in element:
            for tag in element["tags"]:
                if tag.startswith(t):
                    amenity = element["tags"][tag]
                    amenity_counts[amenity] = amenity_counts.get(amenity, 0) + 1
    return amenity_counts

def overpass_bounds(polygon):

    bounds = polygon.bounds
    overpass_bounds = [bounds[1], bounds[0], bounds[3], bounds[2]]
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="school"]({overpass_bounds[0]},{overpass_bounds[1]},{overpass_bounds[2]},{overpass_bounds[3]});
      way["amenity"="school"]({overpass_bounds[0]},{overpass_bounds[1]},{overpass_bounds[2]},{overpass_bounds[3]});
      relation["amenity"="school"]({overpass_bounds[0]},{overpass_bounds[1]},{overpass_bounds[2]},{overpass_bounds[3]});
    );
    out count;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    print(data["elements"][0]["tags"]["total"])
    return data
