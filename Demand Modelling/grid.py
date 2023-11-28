from ModellingUtilities import *
import os
from PlotUtilities import *


# gdf = gpd.read_file('PublicGeoJsons/OtherCities/RotterdamPC4.geojson').to_crs(28992)
# gdf = gdf[~gdf.pc4_code.isin([str(x) for x in [3199, 3198, 3151, 3197, 3191, 3192, 3193, 3194, 3196, 3181, 3190]])]
# townships = gpd.read_file('PublicGeoJsons/OtherCities/townships.geojson').to_crs(28992)
# gdf = townships[townships.name == '\'s-Gravenhage']

if city == 'Rotterdam':
    SA = pd.read_pickle('Service Areas/Rott/Rotterdam --- groot service area').to_crs(28992)
if city == 'Den Haag':
    SA = pd.read_pickle('Service Areas/Den Haag/Den Haag --- 2022').to_crs(28992)
if city == 'AADO':
    SA = pd.read_pickle('Service Areas/Ams/Amsterdam --- Update23032022')
    # SAS = os.listdir('Service Areas/Den Haag')
    # SAS.remove('Den Haag --- No parking Zones')
    # SASL = [pd.read_pickle('Service Areas/Den Haag/' + x).to_crs(28992) for x in SAS]
    # SA =pd.concat(SASL)

def creategrid_SA(gdf, square_size=250):
    from shapely.geometry import Polygon
    import geopandas as gpd
    import numpy as np

    x_min, y_min, x_max, y_max = gdf.total_bounds

    # Align to the grid
    x_min = (np.floor(x_min / square_size) * square_size)
    y_min = (np.floor(y_min / square_size) * square_size)
    x_max = (np.ceil(x_max / square_size) * square_size) + square_size # Extend by one column
    y_max = (np.ceil(y_max / square_size) * square_size) + square_size # Extend by one row

    x_coords = np.arange(x_min, x_max, square_size)
    y_coords = np.arange(y_min, y_max, square_size)

    grid_polygons = []
    index = 0 # Initialize the index

    for y in y_coords[::-1][:-1]: # Reverse y-coordinates here
        for x in x_coords[:-1]:
            grid_polygons.append((index, Polygon([(x, y), (x + square_size, y), (x + square_size, y - square_size), (x, y - square_size)])))
            index += 1 # Increment the index

    # Create the GeoDataFrame with the specified index
    grid = gpd.GeoDataFrame(grid_polygons, columns=['index', 'geometry'], crs=gdf.crs)
    grid.set_index('index', inplace=True)

    return grid
def creategrid_pc4(gdf, square_size=250):
    from shapely.geometry import Polygon
    import geopandas as gpd
    import numpy as np

    x_min, y_min, x_max, y_max = gdf.total_bounds

    # Align to the grid
    x_min = (np.floor(x_min / square_size) * square_size)
    y_min = (np.floor(y_min / square_size) * square_size)
    x_max = (np.ceil(x_max / square_size) * square_size) + square_size # Extend by one column
    y_max = (np.ceil(y_max / square_size) * square_size) + square_size # Extend by one row

    x_coords = np.arange(x_min, x_max, square_size)
    y_coords = np.arange(y_min, y_max, square_size)

    grid_polygons = []
    index = 0 # Initialize the index

    for y in y_coords[::-1][:-1]: # Reverse y-coordinates here
        for x in x_coords[:-1]:
            grid_polygons.append((index, Polygon([(x, y), (x + square_size, y), (x + square_size, y - square_size), (x, y - square_size)])))
            index += 1 # Increment the index

    # Create the GeoDataFrame with the specified index
    grid = gpd.GeoDataFrame(grid_polygons, columns=['index', 'geometry'], crs=gdf.crs)
    grid.set_index('index', inplace=True)

    return grid

#Amsterdam
city = 'AADO'
gdf = getAADO().to_crs(28992)
for square_size in [100]:#[100, 250, 500, 1000]:
    grid = creategrid_pc4(gdf, square_size)
    grid.to_pickle(os.path.join('Demand Modelling', 'Grids', city ,str(square_size)))

#Rotterdam and DH
city = 'Rotterdam'
for square_size in [100]:#[100, 250, 500, 1000]:
    grid = creategrid_SA(SA, square_size)
    grid.to_pickle(os.path.join('Demand Modelling', 'Grids', city, str(square_size)))

#ShowGrid
fig, ax = plt.subplots()
grid.plot(ax = ax, facecolor = 'None', linewidth =0.05)
SA.plot(ax = ax, color = 'Green')
# gdf.plot(ax = ax, facecolor = 'None')




