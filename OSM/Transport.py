from shapely.geometry import Polygon
import json
from ModellingUtilities import getAADO
from PlotUtilities import *
import requests
import geopandas as gpd
from shapely.geometry import Polygon
# city = 'Rotterdam'
square_size = 100

def fetch_data(minlat, minlon, maxlat, maxlon, query):
    overpass_url = "http://overpass-api.de/api/interpreter"

    response = requests.get(overpass_url, params={'data': query})
    if response.status_code == 200:
        data = response.json()
        elements = data['elements']
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [elem['lon'], elem['lat']]
                    },
                    "properties": {
                        "name": elem['tags'].get('name', 'Unknown'),
                        "type": elem['tags'].get('station', elem['tags'].get('amenity', 'Unknown'))
                    }
                } for elem in elements if 'tags' in elem
            ]
        }
        return gpd.GeoDataFrame.from_features(geojson)[['name', 'type', 'geometry']]
    else:
        print(f"Failed to get data from Overpass API, status code: {response.status_code}")
        return None

for city in ['Den Haag', 'AADO', 'Rotterdam']:

    gdf = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size)).to_crs(4326)
    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    amenity_query = f"""
    [out:json];
    (
      node["amenity"]({miny},{minx},{maxy},{maxx});
    );
    out;
    """
    univ_query = f"""
    [out:json];
    (
      node["amenity"="university"]({miny},{minx},{maxy},{maxx});
    );
    out; """
    metro_query = f"""
    [out:json];
    (
      node["railway"="station"]["station"="subway"]({miny},{minx},{maxy},{maxx});
    );
    out;
    """
    tram_query = f"""
    [out:json];
    (
      node["railway"="tram_stop"]({miny},{minx},{maxy},{maxx});
    );
    out;
    """
    dorm_query = f"""
    [out:json];
    (
      node["building"="hall_of_residence"]({miny},{minx},{maxy},{maxx});
      node["amenity"="dormitory"]({miny},{minx},{maxy},{maxx});
      node["access"="students"]({miny},{minx},{maxy},{maxx});
      node["amenity"="student_accommodation"]({miny},{minx},{maxy},{maxx});
      node["building"="dormitory"]({miny},{minx},{maxy},{maxx});
    );
    out; """

    # gdf_dorms = fetch_data(miny, minx, maxy, maxx, dorm_query)
    gdf_univ = fetch_data(miny, minx, maxy, maxx, univ_query)
    gdf_amenities = fetch_data(miny, minx, maxy, maxx, amenity_query)
    gdf_metro = fetch_data(miny, minx, maxy, maxx, metro_query)
    gdf_tram = fetch_data(miny, minx, maxy, maxx, tram_query)
    gdf_horeca = gdf_amenities[gdf_amenities['type'].isin(['restaurant', 'fast_food', 'cafe', 'pub', 'bar'])]


    import os
    for key, value in {'metro':gdf_metro, 'tram': gdf_tram, 'university': gdf_univ ,'amenities': gdf_amenities, 'Horeca': gdf_horeca}.items():
        p =gdf.to_crs(4326).sjoin(value.set_crs(4326)).reset_index().groupby('index').count()['name']
        outgrid = gdf.join(p).to_crs(28992)
        print(key)
        outgrid.plot(column = 'name')
        os.makedirs(os.path.join('OSM', city, str(square_size)), exist_ok = True)
        outgrid.to_pickle(os.path.join('OSM', city, str(square_size), key))


fig, ax = plt.subplots()
for city in ['AADO', 'Rotterdam', 'Den Haag']:
    gdf = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size)).to_crs(4326)
    gdf.plot(ax = ax, facecolor = 'None')
    # value.plot(ax = ax)


for city in ['AADO', 'Rotterdam', 'Den Haag']:
    gdf = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size)).to_crs(4326)

    # Define the bounding box
    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    # Overpass API URL
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Overpass QL query for park areas within the bounding box
    # overpass_query = f"""
    # [out:json];
    # (
    #   way["leisure"="park"]({miny},{minx},{maxy},{maxx});
    #   relation["leisure"="park"]({miny},{minx},{maxy},{maxx});
    # );
    # out geom;
    # """
    #
    overpass_query = f"""
    [out:json];
    (
      way["natural"="water"]({miny},{minx},{maxy},{maxx});
      relation["natural"="water"]({miny},{minx},{maxy},{maxx});
      way["waterway"="riverbank"]({miny},{minx},{maxy},{maxx});
      relation["waterway"="riverbank"]({miny},{minx},{maxy},{maxx});
      way["landuse"="reservoir"]({miny},{minx},{maxy},{maxx});
      relation["landuse"="reservoir"]({miny},{minx},{maxy},{maxx});
      way["waterway"="river"]({miny},{minx},{maxy},{maxx});
      relation["waterway"="river"]({miny},{minx},{maxy},{maxx});
      way["place"="sea"]({miny},{minx},{maxy},{maxx});
      relation["place"="sea"]({miny},{minx},{maxy},{maxx});
    );
    out geom;
    """

    # Fetch the data
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    # Initialize lists to hold polygons and attributes
    polygons = []
    tags_list = []

    # Process ways and build polygons
    for element in data['elements']:
        if element['type'] == 'way':
            points = [(coord['lon'], coord['lat']) for coord in element['geometry']]
            if len(points) > 2:  # Must have at least 3 points to make a polygon
                polygons.append(Polygon(points))
                tags_list.append(element.get('tags', {}))

    # Create GeoDataFrame
    gdf_parks = gpd.GeoDataFrame(tags_list, geometry=polygons, crs="EPSG:4326")
    gdf_parks.plot()
    # Save to shapefile
    # gdf_parks.to_pickle("PublicGeoJsons/Parks/"+ city)

pd.read_pickle('PublicGeoJsons/Water/AmsterdamWater').plot()