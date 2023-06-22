import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import MultiPolygon
import json
file = '/Users/joshuathomas/Desktop/2023-03-01_territories.json'
with open(file, 'r') as f:
  data = json.load(f)[0]
p = data['polygons']

PCs = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
townships = gpd.read_file('PublicGeoJsons/townships.geojson').set_index('name')
am = townships.loc['Amsterdam'].geometry
townships.plot()
l = []
for x in p:
    print(x)
    a = x['coordinates']
    n = []
    for y in a:
        n.append((y['lon'], y['lat']))
    l.append(n)

polygons = [Polygon(coords) for coords in l]
# gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
# gdf.plot()
# ASA =[]
# for i in range(len(gdf)):
#     if gdf.iloc[i].geometry.within(am):
#         ASA.append(gdf.iloc[i].geometry)
# ServiceArea = MultiPolygon(ASA)
# ServiceAreaDF = gpd.GeoDataFrame(geometry=ASA, crs="EPSG:4326")

from geopandas.tools import sjoin
polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
joined_gdf = sjoin(polygons_gdf, townships, how="inner", op='intersects')

ServArea =  joined_gdf[joined_gdf.index_right.isin(['Amsterdam', 'Amstelveen', 'Diemen'])].
fig, ax = plt.subplots()
PCs.plot(ax = ax,facecolor = 'None', linewidth = 0.1)
ServArea.plot(ax = ax)
ServArea.to_pickle('AmsterdamServiceArea')


#Extra
import geopandas as gpd
PC4 = gpd.read_file('PublicGeoJsons/NetherlandsPC4.geojson')
hi = PC4[(PC4.gem_name.isin(['Amsterdam','Amstelveen', 'Diemen', 'Ouder-Amstel']) )| PC4.pc4_code.isin(['1112'])]
hi.plot()
hi['pc4_code']

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
hi.plot(ax = ax, linewidth = 0.3, facecolor = 'None')
for x, y, label in zip(hi.geometry.centroid.x, hi.geometry.centroid.y, hi['pc4_code']):
    ax.text(x, y, str(label), fontsize = 8)

plt.show()




PC4 = PC4[(PC4.gem_name.isin(['Amstelveen', 'Diemen', 'Ouder-Amstel']) )| PC4.pc4_code.isin(['1112'])]
PC4 = PC4[['gem_name', 'geo_point_2d', 'pc4_code', 'geometry']]
PC4.plot()
PC4.to_pickle('PublicGeoJsons/DnA.geojson')

gp =PC4.gem_name.unique()