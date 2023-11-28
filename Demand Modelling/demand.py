from ModellingUtilities import *
from PlotUtilities import *

counts = pd.read_pickle('Misc/centroid_trip_counts_end').rename("count").to_frame()
c = getCentroids().reset_index().set_index('geometry')
AADO_Centroids = counts.join(c).reset_index().set_index('CENTROIDNR').set_geometry('end')

AADO_Centroids.plot()

# betweengridcounts
vmatogriddict(city, square_size):
    grid = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size))
    VMAGrid = grid.sjoin(AADO_Centroids.to_crs(28992))

def felyx_movementgrid(grid):
    f = pd.read_pickle('FelyxData/Raw Movement/Felyx' + city).to_crs(28992)[['geometry', 'prev_location']]
    duplicate_indices = f.index.duplicated(keep='last')
    f = f[~duplicate_indices]
    f = f.set_geometry('prev_location').to_crs(28992)
    pp = gpd.sjoin(f,grid,  how="inner", op="intersects")
    f = f.loc[pp.index]
    k = grid.sjoin(f.set_geometry('prev_location')).index
    h = grid.sjoin(f.set_geometry('geometry')).index
    fmovementgrid = pd.DataFrame({'start': list(h), 'end': list(k)}, index = f.index)
    # hi = movementgrid.groupby(['start', 'end']).size().reset_index(name='count')

    return fmovementgrid
fmovementgrid =felyx_movementgrid(grid)
hi = fmovementgrid.groupby(['start', 'end']).size().reset_index(name='count').set_index(['start', 'end'])


city = 'AADO'
gdf = getAADO()
#gpd.read_file('PublicGeoJsons/RotterdamPC4.geojson')
gdf = gdf.to_crs(28992)
square_size = 250
def gridcounts(square_size, city):
    grid = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size))
    f = pd.read_pickle('FelyxData/Raw Movement/Felyx' + city).to_crs(28992)
    joined = gpd.sjoin(f, grid, how='left', op='within')
    counts = joined.groupby('index_right').size()
    grid['start_counts'] = counts
    grid['start_counts'] = grid['start_counts'].fillna(0)
    # grid[grid.start_counts> 0].plot(column = 'start_counts')
    # grid = grid.dropna()
    # grid = grid[grid['start_counts']> 5]
    return grid
f_grid = gridcounts(square_size, city)



def convolve(VMAGridCount, col = 'count'):
    bounds = VMAGridCount.geometry.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    grid_dims = (int(width / square_size),int(height / square_size))
    r = VMAGridCount[col].fillna(0).values.reshape(grid_dims)
    import numpy as np
    z= np.zeros(grid_dims).reshape(grid_dims)
    for i in range(0, r.shape[0]):
        for j in range(0, r.shape[1]):
            z[i,j] = r[i-1:i+2, j-1:j+2].sum().sum()
    VMAGridCount['convoluted_' + col] = z.reshape(-1)
    return VMAGridCount
f_grid = convolve(f_grid, 'start_counts')
VMAGridCount = convolve(VMAGridCount)

fig, ax = plt.subplots()
#f_grid[f_grid.convoluted_start_counts > 5].plot(ax = ax,  column = 'convoluted_start_counts', legend = True)
f_grid[f_grid.start_counts > 5].plot(ax = ax,  column = 'start_counts', legend = True)
gdf.plot(ax = ax, facecolor = 'None')
gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax = ax, color = 'y')
AADO_Centroids.to_crs(28992).plot(ax = ax, markersize = 5)

def Vma_count(AADO_Centroids, square_size = 250):
    grid = pd.read_pickle('Demand Modelling/Grids/' + str(square_size))
    VMAGrid = grid.sjoin(AADO_Centroids.to_crs(28992))
    VMACount = VMAGrid[['count']].groupby(level = 0).sum()
    VMAGridCount = grid.join(VMACount).fillna(0)
    return VMAGridCount
VMAGridCount = Vma_count(AADO_Centroids, 20)

fig, ax = plt.subplots()
gdf.plot(ax = ax, facecolor = 'None')
# VMAGrid.plot(ax = ax, alpha = 0.2)
VMAGridCount.plot(ax = ax, alpha = 0.8, column = 'count', cmap='OrRd')
# VMAGridCount[VMAGridCount['convoluted_count']> 2500].plot(ax = ax, column = 'convoluted_count')
AADO_Centroids.to_crs(28992).plot(ax = ax, markersize = 1)

