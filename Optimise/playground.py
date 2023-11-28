from ModellingUtilities import *
from PlotUtilities import *
import geopandas as gpd
from shapely.geometry import Polygon

square_size = 200
p  = gpd.read_file('PublicGeoJsons/Parks.json')

gdf1 = getAADO().to_crs(28992)
gdf1 = gpd.read_file('PublicGeoJsons/AmsterdamPC4_nowater.json').to_crs(28992).set_index('Postcode4')
gdf2 = getASA().to_crs(28992)
grid =  pd.read_pickle('Demand Modelling/Grids/AADO/' + str(square_size)).to_crs(28992)

def get_outline(water = True, outline = False):
    gdfA = getAADO(water).to_crs(28992)
    # boundary_gdf = gpd.GeoDataFrame(gdfA.index, geometry=gdfA['geometry'].boundary)
    merged_geometry = gdfA['geometry'].unary_union
    merged_gdf = gpd.GeoDataFrame({'geometry': [merged_geometry]}, columns=['geometry'], crs="EPSG:28992")
    outfile = 'PublicGeoJsons/AADO_Outline'
    if water == True:
        outfile += 'water'

    merged_gdf.to_pickle(outfile)
    merged_gdf.plot(facecolor = 'None', linewidth = 0.25)
    return

get_outline(water = False)

# fig, ax = plt.subplots()
# # p.plot(ax = ax, color = 'Green', alpha = 0.2)
# # ASA.plot(ax = ax, alpha = 0.6)
# hi.plot(ax = ax, column = 'area_intersected')
# # banned_all.to_crs(28992).plot(ax = ax)
# gdf1.to_crs(28992).plot(ax = ax, facecolor = 'None')

def diff(gdf1, gdf2):
    # Create a new GeoDataFrame to store the differences
    differences = gpd.GeoDataFrame(geometry=gdf1.geometry)

    # Use the difference method for each geometry in gdf1 with respect to gdf2
    for idx, geom in gdf1.iterrows():
        difference_geometry = geom['geometry']

        for _, geom2 in gdf2.iterrows():
            difference_geometry = difference_geometry.difference(geom2['geometry'])

        differences.at[idx, 'geometry'] = difference_geometry

    # Save the results if needed
    # differences.to_file("differences.geojson", driver='GeoJSON')

    fig, ax = plt.subplots()
    differences.plot(ax = ax)
    # metro.to_crs(28992).plot(ax = ax, markersize = 1)

    return differences
differences = diff(gdf1, gdf2)

def BannedRoe():
    # Define the bounding box coordinates: [xmin, ymin, xmax, ymax]
    bbox = [485600, 120600, 486700, 122200]  # example bounding box

    # Create a polygon from the bounding box
    polygon = Polygon([(bbox[1], bbox[0]),
                      (bbox[3], bbox[0]),
                      (bbox[3], bbox[2]),
                      (bbox[1], bbox[2])])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': [polygon]})
    gdf.crs = "EPSG:28992"

    bannedRoeter = gdf.overlay(differences)
    return bannedRoeter

def makebannedarea():
    from shapely.geometry import Point
    from shapely.affinity import rotate

    transit = gpd.read_file('PublicGeoJsons/TransitLines.json')
    metro = transit[transit.Modaliteit == 'Metro']
    banned = transit[transit.Naam.isin(['Rembrandtplein', 'Leidseplein', 'Dam'])]

    #9streets, leids, remb
    points = [Point(4.8861, 52.3704), banned.geometry.iloc[0], banned.geometry.iloc[1]]

    banned_gdf = gpd.GeoDataFrame(geometry = points, crs = 4326).to_crs(28992)
    banned_gdf['geometry'] = banned_gdf.buffer(250 / (2**0.5), cap_style=3).envelope
    banned_gdf['geometry'] = banned_gdf['geometry'].apply(lambda geom: rotate(geom, 45, origin='centroid'))
    # banned_gdf.plot()
    # banned_gdf['geometry'].loc[]
    banned_gdf.geometry.iloc[0] = rotate(banned_gdf.geometry.iloc[0], -45, origin='centroid')
    banned_gdf.geometry.iloc[2] = rotate(banned_gdf.geometry.iloc[2], 20, origin='centroid')

    banned_roeter = BannedRoe()
    banned_roeter_9LR = pd.concat([banned_roeter, banned_gdf])
    banned_roeter_9LR_ams1012 = pd.concat([banned_roeter_9LR,gdf1.loc[[1012]]])

    banned_roeter_9LR_ams1012.plot()

    banned_roeter_9LR_ams1012.to_pickle('Misc/BannedAreas')

    return banned_roeter_9LR_ams1012

ba = makebannedarea()

# ADD PARK to this
p = gpd.read_file('PublicGeoJsons/Parks.json').to_crs(28992)
p['park'] = 1
banned_roeter_9LR_ams1012['park'] = 0

banned_all = pd.concat([banned_roeter_9LR_ams1012, p])

bannedSA = pd.read_pickle('Service Areas/Amsterdam --- No parking')
makebannedgrid(100, bannedSA)


def makebannedgrid(square_size, area):
    banned_all = area
    grid = pd.read_pickle('Demand Modelling/Grids/AADO/' + str(square_size)).to_crs(28992)

    # Calculate intersections
    intersected = gpd.overlay(grid, banned_all, how='intersection')
    intersected['area_intersected'] = intersected['geometry'].area
    summed_areas = intersected.groupby('geometry').area_intersected.sum().reset_index()
    summed_areas = gpd.GeoDataFrame(summed_areas, geometry = 'geometry', crs = 28992)
    # summed_areas.plot(column = 'area_intersected', cmap = 'Reds')
    hi = grid.sjoin(summed_areas, how = 'left').fillna(0).groupby('geometry').nth(0)
    hi['area_intersected'] /= grid['geometry'].iloc[0].area
    hi = hi[['geometry', 'area_intersected']]
    hi.plot(column = 'area_intersected', cmap = 'Reds', legend = True)

    hi.to_pickle('BannedGrid' + str(square_size))

    return hi
def makebannedgridpark(square_size):
    banned_all = gpd.read_file('PublicGeoJsons/Parks.json').to_crs(28992)
    grid = pd.read_pickle('Demand Modelling/Grids/AADO/' + str(square_size)).to_crs(28992)

    # Calculate intersections
    intersected = gpd.overlay(grid, banned_all, how='intersection')
    intersected['area_intersected'] = intersected['geometry'].area
    summed_areas = intersected.groupby('geometry').area_intersected.sum().reset_index()
    summed_areas = gpd.GeoDataFrame(summed_areas, geometry = 'geometry', crs = 28992)
    # summed_areas.plot(column = 'area_intersected', cmap = 'Reds')
    hi = grid.sjoin(summed_areas, how = 'left').fillna(0).groupby('geometry').nth(0)
    hi['area_intersected'] /= grid['geometry'].iloc[0].area
    hi = hi[['geometry', 'area_intersected']]
    hi.plot(column = 'area_intersected', cmap = 'Reds', legend = True)

    hi.to_pickle('ParkBannedGrid' + str(square_size))

    return hi
def makebannedgridwater(square_size):
    banned_all = pd.read_pickle('PublicGeoJsons/AmsterdamWater').to_crs(28992)
    grid = pd.read_pickle('Demand Modelling/Grids/AADO/' + str(square_size)).to_crs(28992)

    # Calculate intersections
    intersected = gpd.overlay(grid, banned_all, how='intersection')
    intersected['area_intersected'] = intersected['geometry'].area
    summed_areas = intersected.groupby('geometry').area_intersected.sum().reset_index()
    summed_areas = gpd.GeoDataFrame(summed_areas, geometry = 'geometry', crs = 28992)
    # summed_areas.plot(column = 'area_intersected', cmap = 'Reds')
    hi = grid.sjoin(summed_areas, how = 'left').fillna(0).groupby('geometry').nth(0)
    hi['area_intersected'] /= grid['geometry'].iloc[0].area
    hi = hi[['geometry', 'area_intersected']]
    hi.plot(column = 'area_intersected', cmap = 'Reds', legend = True)

    hi.to_pickle('BannedGrids/WaterBannedGrid' + str(square_size))

    return hi

for i in [100, 150]:
    # makebannedgrid(i, pd.read_pickle('Misc/BannedAreas'))
    # makebannedgridpark(i)
    makebannedgridwater(i)









