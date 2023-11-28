from ModellingUtilities import *
from PlotUtilities import *

polygons = pd.read_pickle('Misc/AADO_VMA_polygons').set_index('index_right')[['geometry']]
polygons.index = polygons.index.rename('Centroid')
ASA_Polygon_Index = list(polygons.sjoin(getASA()[['geometry']]).index)
CentroidBuurt =  pd.read_pickle('VMA/VMA_Cleaning/Clean Data/Centroids/CleanCentroids')[['geometry']].sjoin(gpd.read_file('PublicGeoJsons/AmsBuurts.json')[['Buurt', 'geometry']])[['Buurt']]


# def VMA_by_grid():
#     VMA = pd.read_pickle('VMA/VMA_Cleaning/commondf')
#     VMA['sum'] = VMA.iloc[:, 2:].sum(axis = 1)
#     VMA = VMA[['CENTROIDNR_x', 'CENTROIDNR_y','sum']]
#     VMA = VMA[VMA['CENTROIDNR_x'].isin(ASA_Polygon_Index) & VMA['CENTROIDNR_y'].isin(ASA_Polygon_Index)]
#     VMA = VMA.groupby(['CENTROIDNR_x', 'CENTROIDNR_y']).mean().rename(columns = {'sum' : 'trips'})
#     return VMA
#
# VMA = VMA_by_grid()
#
# square_size = 250
# city = 'AADO'
# grid = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size))
#
# sub = VMA[VMA['CENTROIDNR_x'] == '100']
# centroids = pd.read_pickle('VMA/VMA_Cleaning/Clean Data/Centroids/CleanCentroids')[['geometry']]
# sub['sum'] = sub.iloc[:, 2:].sum(axis = 1)
# sub = sub[['CENTROIDNR_x', 'CENTROIDNR_y','sum']]
#
# sub = sub.set_index('CENTROIDNR_y').join(centroids)
# sub = gpd.GeoDataFrame(sub, crs = 4326, geometry = 'geometry').to_crs(28992)
# subg = grid.sjoin(sub)
#
# subg.plot(column = 'sum', cmap = 'Reds', vmax = 25)
#
#
# centroids.join(sub.set_index('CENTROIDNR_y'))




def felyx_by_centroid():
    city = 'AADO'
    felyx2022 = pd.read_pickle('FelyxData/Raw Movement/Felyx' + city + '2022')#.to_crs(28992)
    felyx2023 = pd.read_pickle('FelyxData/Raw Movement/Felyx' + city)#.to_crs(28992)
    felyx = pd.concat([felyx2022, felyx2023])
    felyx = felyx.set_geometry('prev_location')[['geometry', 'prev_location']]
    felyx = felyx.sjoin(polygons).set_geometry('geometry')
    felyx = felyx.rename(columns = {'index_right' : 'start_centroid'})[['start_centroid', 'geometry']]
    felyx = felyx.sjoin(polygons)
    felyx = felyx.rename(columns = {'index_right' : 'end_centroid'})
    felyx = felyx.rename(columns={'start_centroid': 'CENTROIDNR_x', 'end_centroid': 'CENTROIDNR_y'})
    felyx = felyx.groupby(['CENTROIDNR_x', 'CENTROIDNR_y']).count().rename(columns = {'geometry' : 'trips'})

    return felyx

def VMA_by_centroid():
    VMA = pd.read_pickle('VMA/VMA_Cleaning/commondf')
    VMA['sum'] = VMA.iloc[:, 2:].sum(axis = 1)
    VMA = VMA[['CENTROIDNR_x', 'CENTROIDNR_y','sum']]
    VMA = VMA[VMA['CENTROIDNR_x'].isin(ASA_Polygon_Index) & VMA['CENTROIDNR_y'].isin(ASA_Polygon_Index)]
    VMA = VMA.groupby(['CENTROIDNR_x', 'CENTROIDNR_y']).mean().rename(columns = {'sum' : 'trips'})
    return VMA

def Centroid_Outbound(df):
    OJ = df.reset_index().set_index('CENTROIDNR_y')
    OJ = polygons.join(OJ)
    OJ = OJ.set_index('CENTROIDNR_x')
    return OJ

VMA_out = Centroid_Outbound(VMA_by_centroid())
felyx_out = Centroid_Outbound(felyx_by_centroid())

def plot_outbound(x, m = 0.0005):

    fig, ax = plt.subplots()
    polygons.loc[ASA_Polygon_Index].plot(ax = ax, facecolor = 'None', linewidth = 0.05)
    to_plot = felyx_out.loc[str(x)]
    to_plot.trips /= sum(to_plot.trips)
    to_plot.plot(column = 'trips', ax = ax, alpha = 1, legend = True)
    polygons.loc[[x]].plot(ax=ax, facecolor='Red', linewidth=1)
    gpd.read_file('PublicGeoJsons/AmsLines.json').plot(ax = ax, color = 'Yellow', linewidth = 0.5)
    ax.set_title('felyx' + ' : ' + CentroidBuurt.loc[x].values[0])

    fig, ax2 = plt.subplots()
    polygons.loc[ASA_Polygon_Index].plot(ax = ax2, facecolor = 'None', linewidth = 0.05)
    to_plot = VMA_out.loc[str(x)]
    to_plot.trips /= sum(to_plot.trips)
    to_plot = to_plot[to_plot.trips > m]
    to_plot.plot(column = 'trips', ax = ax2, alpha = 1, legend = True)
    polygons.loc[[x]].plot(ax=ax2, facecolor='Red', linewidth=1)
    # Transit.plot(markersize = 3, color = 'yellow', ax = ax2)
    gpd.read_file('PublicGeoJsons/AmsLines.json').plot(ax=ax2, color='Yellow', linewidth=0.5)
    ax2.set_title('VMA'+ ' : ' + CentroidBuurt.loc[x].values[0])
    return to_plot

view = plot_outbound(ASA_Polygon_Index[4])


i = 1280
h = fweighted_journey_df.felyx_weighted_summed_dict.iloc[i]
hi = fweighted_journey_df.index[i]

goto = pd.DataFrame(list(h.items()), columns=['Centroid', 'Value']).set_index('Centroid')
goto['Value'] /= 1000

fig, ax = plt.subplots()
polygons.join(goto).dropna().plot(ax = ax, column = 'Value', legend = 'True')
polygons.loc[ASA_Polygon_Index].plot(ax = ax, facecolor = 'None', linewidth = 0.05)
grid.to_crs(4326).loc[[hi]].plot(ax=ax, facecolor='Red', linewidth=1)
