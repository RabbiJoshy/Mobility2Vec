import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from geopandas.tools import sjoin
from shapely import MultiPolygon
import json
from ModellingUtilities import getAADO
['Amsterdam','Amstelveen', 'Diemen', 'Ouder-Amstel']
file1 = '/Users/joshuathomas/Desktop/2023-03-01_territories.json'
file = '/Users/joshuathomas/Desktop/2023-06-01_territories.json'



AADO = getAADO()
pc4_nowater = gpd.read_file('PublicGeoJsons/AmsterdamPC4_nowater.json').set_index('Postcode4')
pc4_nowater.index = pc4_nowater.index.astype(str)
AADO.loc[pc4_nowater.index] = pc4_nowater


AADO = getAADO()
AADO_water = getAADO(water = True)
difference_result = AADO_water.geometry.difference(AADO.unary_union)
AADO_difference = gpd.GeoDataFrame(geometry=difference_result)
AADO_difference.to_pickle('PublicGeoJsons/AmsterdamWater')


def get_service_area(file, cities = ['Amsterdam', 'Amstelveen', 'Diemen']):
    with open(file, 'r') as f:
        data = json.load(f)[0]
    p = data['polygons']
    PCs = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
    townships = gpd.read_file('PublicGeoJsons/townships.geojson').set_index('name')
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
Rotterdam = get_service_area(file, ['\'s-Gravenhage'])
# Rotterdam.to_pickle('Misc/DHServiceArea')


with open(file1, 'r') as f:
    data = json.load(f)[0]
p = data['polygons']
l = []
for x in p:
    print(x)
    # print(x)
    a = x['coordinates']
    n = []
    for y in a:
        n.append((y['lon'], y['lat']))
    l.append(n)
polygons = [Polygon(coords) for coords in l]
polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
polygons_gdf.plot()






#Extra
def DiemenFiles():
# Gem = gpd.read_file('PublicGeoJsons/Gemeenten.geojson')
# AADO_Gem = Gem[Gem.statnaam.isin(['Amsterdam','Amstelveen', 'Diemen', 'Ouder-Amstel'])]
# ADO_Gem = Gem[Gem.statnaam.isin(['Amstelveen', 'Diemen', 'Ouder-Amstel'])]
# AADO_Gem = AADO_Gem.to_crs('4326')
# ADO_Gem = ADO_Gem.to_crs('4326')

    PC4 = gpd.read_file('PublicGeoJsons/NetherlandsPC4.geojson').set_index('PC4')
    # AADO = gpd.sjoin(PC4, AADO_Gem, how='inner', op='intersects')
    # ADO = gpd.sjoin(PC4, ADO_Gem, how='inner', op='intersects')
    ADO.plot()
    AADO.plot()
    AADO_Gem.plot()
    AADOPOSTCODES = list(range(1180, 1192)) + list(range(1011, 1116)) + list(range(1380, 1385))
    AADOPOSTCODES = [str(x) for x in AADOPOSTCODES if str(x) in PC4.index]
    AADO_PC4 = PC4.loc[AADOPOSTCODES]

    ADOPOSTCODES = list(range(1180, 1192)) + list(range(1110, 1116))
    ADOPOSTCODES = [str(x) for x in ADOPOSTCODES if str(x) in PC4.index]
    ADO_PC4 = PC4.loc[ADOPOSTCODES]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ADO_PC4.plot(ax = ax, linewidth = 0.3, facecolor = 'None')
    for x, y, label in zip(ADO_PC4.geometry.centroid.x, ADO_PC4.geometry.centroid.y, ADO_PC4.index):
        ax.text(x, y, str(label), fontsize = 8)
    gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson').plot(ax = ax)

    AADO_PC4.to_pickle('PublicGeoJsons/AADO_PC4.geojson')
    ADO_PC4.to_pickle('PublicGeoJsons/ADO_PC4.geojson')

    return


import pandas as pd
Odin = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021All')
# AOdin = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021Ams')
ODINAADO = Odin[Odin.aankpc.isin(AADO_PC4.index) & Odin.vertpc.isin(AADO_PC4.index)]
ODINAADO .to_pickle('Odin/OdinWrangled/Odin2018-2021ADO')
