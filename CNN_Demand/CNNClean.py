from ModellingUtilities import *
from PlotUtilities import *

counts = pd.read_pickle('Misc/centroid_trip_counts_end').rename("count").to_frame()
c = getCentroids().reset_index().set_index('geometry')
AADO_Centroids = counts.join(c).reset_index().set_index('CENTROIDNR').set_geometry('end')
# AADO_Centroids.plot()

trans = gpd.read_file('PublicGeoJsons/Transport/TransitLines.json').to_crs(28992)
trains = pd.read_csv('PublicGeoJsons/Transport/NS/stations-2022-01-nl.csv')
trains = gpd.GeoDataFrame(
    trains, geometry=gpd.points_from_xy(trains.geo_lng, trains.geo_lat), crs="EPSG:4326"
).to_crs(28992)[['geometry', 'name_short']]
metros = trans[trans.Modaliteit == 'Metro'][['geometry', 'Naam']]

city = 'AADO'
gdf = getAADO()
#gpd.read_file('PublicGeoJsons/RotterdamPC4.geojson')
gdf = gdf.to_crs(28992)
square_size = 150
felyx = pd.read_pickle('FelyxData/Raw Movement/Felyx' + city).to_crs(28992)

def categorize_time(df, datetime_column): #TODO move to wrangler
    # Convert the datetime column to pandas datetime if it's not already
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Create a new column to store the time category
    df['time_category'] = 'Other'

    # Define the time ranges
    morning_start = pd.to_datetime('07:00:00').time()
    morning_end = pd.to_datetime('09:00:00').time()
    daytime_start = pd.to_datetime('09:00:00').time()
    daytime_end = pd.to_datetime('16:00:00').time()
    evening_start = pd.to_datetime('16:00:00').time()
    evening_end = pd.to_datetime('18:00:00').time()

    # Categorize the time for each row
    df.loc[(df[datetime_column].dt.time >= morning_start) & (df[datetime_column].dt.time <= morning_end), 'time_category'] = 'Morning'
    df.loc[(df[datetime_column].dt.time > evening_start) & (df[datetime_column].dt.time < evening_end), 'time_category'] = 'Evening'
    df.loc[(df[datetime_column].dt.time >= daytime_start) & (df[datetime_column].dt.time <= daytime_end), 'time_category'] = 'Daytime'

    return df
felyx = categorize_time(felyx, 'prev_time')


grid = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size))
def gridcounts(grid, feature, col_name):
    joined = gpd.sjoin(feature, grid, how='left', op='within')
    counts = joined.groupby('index_right').size()
    grid['counts_' + col_name] = counts
    grid['counts_' + col_name] = grid['counts_' + col_name].fillna(0)
    return grid
f_grid = gridcounts(grid, felyx, 'fstart')
trains_grid = gridcounts(f_grid, trains, 'train')
metro_grid = gridcounts(trains_grid, metros, 'metro')

pc4 = pd.read_pickle('PostcodeInfo/PC4_Clean')
pc4 = pc4.rename(columns = {'Unnamed: 36': 'density'})
pc4 = pc4[['Man', 'density']]
density = gpd.read_file('PublicGeoJsons/NetherlandsPC4.geojson')
density = density.set_index('PC4')
density = density.to_crs(28992)[['geometry']]
infopc4 = density.join(pc4, how = 'inner')


# df3 = gpd.overlay(grid, infopc4, how='intersection')
df3 = grid[['geometry']].sjoin(infopc4, how='inner')
# df3.plot()
#
# df3['area'] = df3.geometry.area
# df3.sort_values(by='area', inplace=True)

df3 = df3[~df3.index.duplicated(keep='first')]

cnngrid = grid.join(df3.drop('geometry', axis =1)).fillna(0)
Amenities = pd.read_pickle('SAO/OSM/Amenities/AmenitiesGrid' + str(square_size))
# v = Amenities.sum().sort_values(ascending = False)[:20]
Horeca = Amenities[['restaurant', 'fast_food', 'cafe', 'pub', 'bar']]
Horeca['Horeca'] = Horeca.sum(axis = 1)
cnngrid = cnngrid.join(Horeca['Horeca'])

cnngrid.plot(column = 'Horeca')
cnngrid.plot(column = 'counts_train')
cnngrid.to_pickle('CNN Demand/CNN_Data/' + str(square_size))




fig, ax = plt.subplots()
#f_grid[f_grid.convoluted_start_counts > 5].plot(ax = ax,  column = 'convoluted_start_counts', legend = True)
cnngrid.plot(ax = ax,  column = 'Man', legend = True)
gdf.to_crs(28992).plot(ax = ax, facecolor = 'None')
gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax = ax, color = 'y')
AADO_Centroids.to_crs(28992).plot(ax = ax, markersize = 5)