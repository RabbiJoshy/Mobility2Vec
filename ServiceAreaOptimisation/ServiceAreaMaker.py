import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import json
import os
import pandas as pd
from ModellingUtilities import getAADO
#['Amsterdam','Amstelveen', 'Diemen', 'Ouder-Amstel']
file = '/Users/joshuathomas/Desktop/2023-03-01_territories.json'
file = 'Misc/all.json'

month = '03'
file = 'Service Areas/Ams/2023-03-01_territories.json'

def get_service_area(file, cities = ['Amsterdam', 'Amstelveen', 'Diemen']):
    with open(file, 'r') as f:
        data = json.load(f)[0]
    p = data['polygons']
    PCs = pd.read_pickle('PublicGeoJsons/AADO_PC4.geojson')
    # PCs = gpd.read_file('PublicGeoJsons/AADO_PC4.geojson')
    townships = gpd.read_file('PublicGeoJsons/OtherCities/townships.geojson').set_index('name')
    # am = townships.loc['Amsterdam'].geometry
    # townships.plot()
    l = []
    for x in p:
        # print(x)
        a = x['coordinates']
        n = []
        for y in a:
            n.append((y['lon'], y['lat']))
        l.append(n)
    polygons = [Polygon(coords) for coords in l]

    from geopandas.tools import sjoin
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    joined_gdf = sjoin(polygons_gdf, townships, how="inner", op='intersects')

    print(joined_gdf.index_right.unique())

    ServArea =  joined_gdf[joined_gdf.index_right.isin(cities)]
    fig, ax = plt.subplots()
    PCs.plot(ax = ax,facecolor = 'None', linewidth = 0.1)
    ServArea.plot(ax = ax)
    # ServArea.to_pickle('AmsterdamServiceArea')

    return ServArea
SA03 = get_service_area(file)
SA03.to_pickle('Service Areas/Ams/2023-03-01')
# Rotterdam = get_service_area(file, ['\'s-Gravenhage'])
# Rotterdam.to_pickle('Misc/DHServiceArea')


def get_service_area_from_all_file(city = 'Ams'):

    with open(file, 'r') as f:
        data = json.load(f)
        cityinfo = [(i['title'], i) for i in data if i['title'][:len(city)] == city]

        maps = {}
        for title, info in cityinfo:
            print(title)
            polygoninfo = info['polygons']

            coordinates = []
            for info in polygoninfo:
                a = info['coordinates']
                poly_coords = []
                for y in a:
                    poly_coords.append((y['lon'], y['lat']))
                coordinates.append(poly_coords)
            polygons = [Polygon(coords) for coords in coordinates]
            polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4236").to_crs(28992)
            # polygons_gdf.plot()
            maps[title] = polygons_gdf

    return maps

city = 'Gron'
maps = get_service_area_from_all_file(city)
for key in maps.keys():
    print(key)
    os.makedirs(os.path.join('Service Areas', city), exist_ok=True)
    maps[key].to_pickle(os.path.join('Service Areas', city, key))
    maps[key].plot()


