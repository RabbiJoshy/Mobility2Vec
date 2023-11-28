from ModellingUtilities import *
from PlotUtilities import *
from itertools import product

square_size = 150
city = 'AADO'

def groupfelyxbystartend(a, b):
       F = pd.read_pickle('FelyxData/Raw Movement/FelyxAADO')
       F = F[F.prev_time.dt.hour.between(a, b)]
       poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).rename(columns = {'index_right': 'end_centroid'}).set_index('end_centroid')[['geometry']]
       # c =pd.read_pickle('VMA/VMA_Cleaning/Clean Data/Centroids/CleanCentroids')
       df = F[['prev_location', 'geometry']].to_crs(28992).sjoin(poly).rename(columns = {'index_right': 'end_centroid'})
       df = df.set_geometry('prev_location').to_crs(28992)
       df = df[['prev_location', 'end_centroid']].sjoin(poly).rename(columns = {'index_right': 'start_centroid'})
       felyx_VMA_start_end = df.groupby([ 'start_centroid', 'end_centroid']).count()
       felyx_VMA_start_end.rename(columns = {'prev_location': 'count'}, inplace= True)
       felyx_VMA_start_end = felyx_VMA_start_end.reset_index().set_index('start_centroid')
       felyx_VMA_start_end.to_pickle('EndLocation/felyx_VMA_start_end')
       return felyx_VMA_start_end
Morning = groupfelyxbystartend(6, 8)

# F = pd.read_pickle('EndLocation/felyx_VMA_start_end')

def add_reverse_count(df):
       merged_df = df.reset_index().merge(
           df.reset_index(),
           left_on=['start_centroid', 'end_centroid'],
           right_on=['end_centroid', 'start_centroid'],
           suffixes=('', '_reverse')
       )

       # Select relevant columns and rename them
       merged_df = merged_df[['start_centroid', 'end_centroid', 'count', 'count_reverse']]
       merged_df.columns = ['start_centroid', 'end_centroid', 'count', 'reverse_count']

       # Update the original DataFrame with the 'reverse_count' information
       df = df.merge(
           merged_df[['start_centroid', 'end_centroid', 'reverse_count']],
           how='left',
           left_on=['start_centroid', 'end_centroid'],
           right_on=['start_centroid', 'end_centroid']
       ).fillna(0)
       return df
df = add_reverse_count(F)
all_centroids = pd.read_pickle('Misc/AADO_VMA_polygons').index_right.values
VMA = pd.read_pickle('quick_VMA_preds_new')
centroids =pd.read_pickle('VMA/VMA_Cleaning/Clean Data/Centroids/CleanCentroids')[['geometry']].reset_index()

df.set_index('start_centroid', inplace = True)
useful = []
for i in df.index.unique():
    # print(len(df.loc[i]))
    if(len(df.loc[i]) >= 0):
        useful.append(i)
df = df.loc[useful]

useful = all_centroids

all_combinations = pd.DataFrame(list(product(useful, repeat=2)), columns=['start_centroid', 'end_centroid'])
final_df = pd.DataFrame()
for start in useful:
    print(start)
    # Create a filtered DataFrame where 'start_centroid' is the current 'start'
    filtered_combinations = all_combinations[all_combinations['start_centroid'] == start]

    # Merge with original DataFrame to get counts, fill NaN with 0
    new_df = filtered_combinations.merge(df, how='left', on=['start_centroid', 'end_centroid']).fillna(0)

    # Append to final DataFrame
    final_df = pd.concat([final_df, new_df])
final_df.to_pickle('EndLocation/final_df' + str(square_size))

finaldf = pd.read_pickle('EndLocation/final_df')
# Merge centroid locations into the DataFrame
final_df = pd.merge(all_combinations, centroids, left_on='start_centroid', right_on='CENTROIDNR', how='left')
final_df.rename(columns={'geometry': 'start_geometry'}, inplace=True)
final_df = pd.merge(final_df, centroids, left_on='end_centroid', right_on='CENTROIDNR', how='left')
final_df.rename(columns={'geometry': 'end_geometry'}, inplace=True)
final_df['distance'] = final_df.apply(lambda row: row['start_geometry'].distance(row['end_geometry']), axis=1)
final_df = final_df.merge(df, on=['start_centroid', 'end_centroid'], how='left').fillna(0)
final_df.to_pickle('EndLocation/ModellingData' +  str(square_size))

final_df = pd.read_pickle('EndLocation/ModellingData' +  str(square_size))
VMA = pd.read_pickle('quick_VMA_preds_new')
VMAlite = VMA[['CENTROIDNR_x', 'CENTROIDNR_y', 'num_trips']].groupby(['CENTROIDNR_x', 'CENTROIDNR_y']).mean()
final_with_vma = final_df.set_index(['CENTROIDNR_x', 'CENTROIDNR_y']).join(VMAlite).fillna(0)
final_with_vma.to_pickle('EndLocation/final_df_withvma' + str(square_size))
modelling_data = pd.read_pickle('EndLocation/final_df_withvma' + str(square_size))


modelling_data['normalized_num_trips'] = modelling_data['num_trips'] / modelling_data.groupby('start_centroid')['num_trips'].transform('sum')
modelling_data['normalized_count'] = modelling_data['count'] / modelling_data.groupby('start_centroid')['count'].transform('sum')
modelling_data['normalized_reverse_count'] = modelling_data['reverse_count'] / modelling_data.groupby('end_centroid')['reverse_count'].transform('sum')
# modelling_data[modelling_data['start_centroid'] =='100'][['distance', 'count', 'reverse_count', 'num_trips',
#        'normalized_num_trips', 'normalized_count', 'normalized_reverse_count']].sum()
modelling_data.to_pickle('EndLocation/modelling_data' +  str(square_size))



modelling_data = pd.read_pickle('EndLocation/modelling_data' + str(square_size)).fillna(0)[['start_centroid', 'end_centroid', 'end_geometry',
       'distance', 'count', 'reverse_count', 'num_trips', 'normalized_num_trips', 'normalized_count', 'normalized_reverse_count']]
infopc4 = pd.read_pickle('PostcodeInfo/PC4_Clean_Geo')[['density', 'geometry']]
geomod = gpd.GeoDataFrame(modelling_data, crs = 4326, geometry = 'end_geometry').to_crs(28992)


df3 = geomod.sjoin(infopc4.set_crs(28992), how='inner')
df3 = df3[~df3.index.duplicated(keep='first')]
df3.to_pickle('EndLocation/geomodelling_data' + str(square_size))

geo_modelling_data = pd.read_pickle('EndLocation/geomodelling_data'+ str(square_size))
a = geo_modelling_data.sample(10000)

import xgboost as xgb
model = xgb.XGBRegressor()
from sklearn.model_selection import train_test_split
train, test, = train_test_split(geo_modelling_data, test_size=0.2, random_state=42)
features = ['normalized_reverse_count', 'distance', 'normalized_num_trips', 'density']
model.fit(train[features], train['normalized_count'].values)
y_pred = model.predict(test[features])
from sklearn.metrics import mean_squared_error
mean_squared_error(test['normalized_count'].values, y_pred)
test['pred'] = y_pred
test.to_pickle('EndLocation/Preds' + str(square_size))

# test = pd.read_pickle('EndLocation/Preds' + str(square_size))
predictions_all = model.predict(geo_modelling_data[features])
geo_modelling_data['pred'] = predictions_all
geo_modelling_data = geo_modelling_data[['start_centroid', 'end_centroid', 'end_geometry', 'pred']]
geo_modelling_data.to_pickle('EndLocation/Predictions_All'+ str(square_size))


#START BY NORMALISING THE PROPORTIONS (ALL OF THEM, to get rid of negatives but also so that they all add to 1)

predictions_all = pd.read_pickle('EndLocation/Predictions_All' + str(square_size))
shift_val = abs(predictions_all['pred'].min()) + 0.001  # Adding a small constant
predictions_all['pred_shifted'] = predictions_all['pred'] + shift_val
group_sums = predictions_all.groupby('CENTROIDNR_x')['pred_shifted'].transform('sum')
predictions_all['pred_normalised'] = predictions_all['pred_shifted'] / group_sums
predictions_all.to_pickle('EndLocation/Predictions_All' + str(square_size))

VMA = pd.read_pickle('quick_VMA_preds_new')
VMAlite = VMA[['CENTROIDNR_x', 'CENTROIDNR_y', 'num_trips']].groupby(['CENTROIDNR_x', 'CENTROIDNR_y']).mean()

###PREDICTIONS KNWON IS THE PREDICTED NORMALISED) PROPRORTION  OF JOURNEYS BETWEEN ALL! CENTROIDS -
predictions_known = predictions_all.loc[list(set(VMAlite.index).intersection(set(predictions_all.index)))]
#SHOULD I DO THIS BEFORE OR BEFOIRE AND AFTER THE KNOWN BIT
# DOING IT AFTER MEANS EACH CENTROID GETS 1 OUTWARD JOURNEY PER JOURNEY BUT IT WOULD BE BETTER IF I HAD MORE KNOWNS
group_sums = predictions_known.groupby('CENTROIDNR_x')['pred_normalised'].transform('sum')
predictions_known['pred_known_renormalised'] = predictions_known['pred_normalised'] / group_sums
predictions_known.to_pickle('EndLocation/Predictions_Optimiser'  + str(square_size))


from ModellingUtilities import *
from PlotUtilities import *
square_size = 150
predictions_known = pd.read_pickle('EndLocation/Predictions_Optimiser' + str(square_size))
predictions_all = pd.read_pickle('EndLocation/Predictions_All' + str(square_size))

# fig, ax = plt.subplots()
# poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).rename(columns={'index_right': 'end_centroid'}).set_index(
#         'end_centroid')[['geometry']]
# poly.plot(ax=ax, facecolor = 'None', linewidth = 0.05)
# cx.add_basemap(ax, crs=poly.crs.to_string(), source=cx.providers.Stamen.Watercolor)
#
# # Annotate each polygon with the index
# for x, y, label in zip(poly.geometry.centroid.x, poly.geometry.centroid.y, poly.index):
#     ax.text(x, y, str(label), fontsize = 8)
#
# plt.show()

def view_predict(centroid, predictions, pred_column = 'pred_known_renormalised'):
    t100 =predictions.set_index('start_centroid').loc[[centroid]]
    poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).rename(columns={'index_right': 'end_centroid'}).set_index(
        'end_centroid')[['geometry']]
    t100poly = pd.merge(t100, poly.reset_index(), left_on='end_centroid', right_on='end_centroid', how='left')

    print(t100poly[pred_column].sum())

    fig, ax = plt.subplots()
    # getAADO().to_crs(28992).plot(ax = ax)
    geo = gpd.GeoDataFrame(t100poly, geometry = 'geometry', crs = 28992)
    geo.plot(column = pred_column, ax = ax, legend = True)
    poly.loc[[centroid]].plot(ax = ax, edgecolor = 'red', facecolor = 'None', linewidth = 2.5)
    cx.add_basemap(ax, crs= geo.crs.to_string(), alpha = 0.5)
    remove_labels(ax)
    return
view_predict('395', predictions_all, 'pred_normalised')
view_predict('653', predictions_known)



len(predictions_all.start_centroid.unique())
b = predictions_known.start_centroid.unique()
len(VMAlite.reset_index().CENTROIDNR_y.unique())

VMA = pd.read_pickle('quick_VMA_preds_new')
VMA['pct_car'] = VMA['Pred_Car']/VMA['num_trips']
gpd.GeoDataFrame(VMA[['start', 'pct_car']].groupby('start').mean().reset_index(), crs = 4326, geometry = 'start').plot(column = 'pct_car', markersize = 2, legend = True)
# import contextily as cx
play = predictions_known.reset_index()[['CENTROIDNR_x' ,'CENTROIDNR_y', 'pred_known_renormalised']]
play = play.merge(VMA[['CENTROIDNR_x' ,'CENTROIDNR_y', 'pct_car']], on = ['CENTROIDNR_x' ,'CENTROIDNR_y'])
play['car_weighted'] = play['pct_car'] * play['pred_known_renormalised'] *100

connsdict = {}
for start in play.CENTROIDNR_x.unique():
    startdf = play[play['CENTROIDNR_x'] == start]
    startconns = dict(zip(startdf['CENTROIDNR_y'], startdf['car_weighted']))
    connsdict[start] = startconns
connsdf = pd.DataFrame(list(connsdict.items()), columns=['CENTROIDNR_x', 'connections_car_weighted'])

connsdf.to_pickle('VMA_Connections' + str(square_size))

# CONNS NOW IMPROVED


conns = pd.read_pickle('VMA_Connections' + str(square_size))
VMA = pd.read_pickle('quick_VMA_preds_new')
VMA['pct_car'] = VMA['Pred_Car']/VMA['num_trips']
























#THIS IS NOT USED ANYNIORE?

df = F.to_crs(28992).sjoin(grid)[['prev_location', 'carId', 'geometry', 'index_right']].rename(columns = {'index_right': 'g_end'})
df.set_geometry('prev_location', inplace = True)
df = df.to_crs(28992).sjoin(grid.to_crs(28992)).rename(columns = {'index_right': 'g_start'})
df = df[['g_start', 'g_end']]

df.to_pickle('Felyxends')

ends_grouped = df.groupby('g_end').count()
ends_grouped = grid.join(ends_grouped)
ends_grouped.plot(column = 'g_start')
ends_grouped= ends_grouped.sort_values(by = 'g_start', ascending = False)

# fig, ax = plt.subplots()
# gn = ends_grouped.index[2]
# grid.join(df[df['g_start']== gn].groupby('g_end').count()).fillna(0).plot(column = 'g_start', cmap = 'Reds', ax = ax)
# grid.loc[[gn]].plot( ax = ax)
# getAADO().to_crs(28992).plot(facecolor = 'None', ax = ax)


felyxtoVMA
polygons = pd.read_pickle('Misc/AADO_VMA_polygons')
h= grid.sjoin(polygons.to_crs(28992).set_index('index_right')).reset_index().groupby('geometry').nth(0)
VMA = pd.read_pickle('VMA/VMA_Cleaning/commondf')
b = VMA.sample(1000)
b.columns
b['trips'] = b[['num_trips_OV_OS', 'num_trips_Fiets_AS',
       'num_trips_OV_RD', 'num_trips_car_AS', 'num_trips_Fiets_RD',
       'num_trips_car_OS', 'num_trips_Fiets_OS', 'num_trips_car_RD',
       'num_trips_OV_AS']].sum(axis = 1)
b = b[['CENTROIDNR_x', 'CENTROIDNR_x', 'trips']]
# b = b.sort_values(by = ['CENTROIDNR_x', 'CENTROIDNR_y'])

