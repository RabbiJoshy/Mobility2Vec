import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas.tools import sjoin
from geopandas.tools import overlay
asa = pd.read_pickle('AmsterdamServiceArea')
tl = gpd.read_file('PublicGeoJsons/TransitLines.json')
lines = gpd.read_file('PublicGeoJsons/AmsLines.json')
AP = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson').set_index('Postcode4')
AP2 = pd.read_pickle('PublicGeoJsons/AADO_PC4.geojson')

AP.index = AP.index.astype(str)
# AP['centroid'] = AP['geometry'].centroid
# AP['centroid_within'] = AP['centroid'].apply(lambda centroid: asa.geometry.contains(centroid).any())
# AP2['centroid_within'] = AP2['centroid'].apply(lambda centroid: asa.geometry.contains(centroid).any())
PCs = gpd.read_file('PublicGeoJsons/Amsbuurts.json')
odin = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021All')
odinams = odin[(odin.aankpc.isin(AP.index)) &(odin.vertpc.isin(AP.index))]
DIEMEN = pd.read_pickle('PublicGeoJsons/ADO_PC4.geojson')
odinams2 = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021AADO')

def countfrac(df, postcodes, ServiceArea):
    postcodes['centroid'] = postcodes['geometry'].centroid
    postcodes['centroid_within'] = postcodes['centroid'].apply(lambda centroid: ServiceArea.geometry.contains(centroid).any())
    fracs = dict()
    for i in ['aankpc', 'vertpc']:
        g = df.groupby(i).count().iloc[:, 0]
        num = g[g.index.isin(postcodes[postcodes['centroid_within'] == True].index)].sum()
        den = g[g.index.isin(postcodes[postcodes['centroid_within'] == False].index)].sum()
        fracs[i] = "{}/{}".format(num, den+num)#round(num/(den+num), 2)
    return fracs
def plotjourneys(df, background, ServiceArea, modechoice = 'Personenauto - bestuurder', samples = 0,
                 choicecol = 'khvm', time = False, transit = False):
    best = df[df[choicecol] == modechoice]
    # best = best.set_index('vertpc').join(best.groupby('vertpc').count().iloc[:, :1].rename(columns={'walk_distance': 'vertcount'}))
    # background = background.join(
    #     best.groupby('aankpc').count().iloc[:, :1].rename(columns={'walk_distance': 'aankcount'}))
    fig, ax = plt.subplots()
    DIEMEN.plot(ax = ax, facecolor = 'y', alpha = 0.1, linewidth = 0.1)
    if transit == True:
        lines.plot(ax = ax, linewidth = 0.5)
        tl['Marker_Size'] = tl['Modaliteit'].map({'Tram': 0.25, 'Metro': 5})
        tl['Color'] = tl['Modaliteit'].map({'Tram': 'y', 'Metro': 'y'})
        tl.plot(ax=ax, c=tl['Color'], markersize=tl['Marker_Size'], legend=True)
    background.plot(ax = ax, linewidth = 0.1,  facecolor = 'None')
    ServiceArea.plot(ax=ax, linewidth=0.1, alpha = 0.5)
    # ax.set_aspect('equal')
    if samples > 0:
        best = best.sample(min(len(best), samples))
    fracdict = countfrac(best, background, ServiceArea)
    asize = best.groupby('aankpc').count().iloc[:, 0]
    vsize = best.groupby('vertpc').count().iloc[:, 0]
    x2 = background.loc[best['aankpc']].geometry.centroid.x
    y2 = background.loc[best['aankpc']].geometry.centroid.y
    x1 = background.loc[best['vertpc']].geometry.centroid.x
    y1 = background.loc[best['vertpc']].geometry.centroid.y
    # ax.scatter(x1, y1, c = 'r', s = vsize[best['vertpc']])
    # ax.scatter(x2, y2, c='g', s=asize[best['aankpc']])
    ax.set_title(str(fracdict))

    return
def Buffer(asa, b = 50):
    asa = asa.to_crs('EPSG:3857')
    asa_buffered = asa.copy()
    asa_buffered['geometry'] = asa.geometry.buffer(b)  # buffer distance of 500m
    asa_buffered = asa_buffered.to_crs('EPSG:4326')
    asa_buffered['dissolve_field'] = 1

    # Use the dissolve method to merge all polygons
    non_overlapping = asa_buffered.dissolve(by='dissolve_field')

    return non_overlapping

asa_buffered = Buffer(asa, 250)
# asa_buffered.plot()
# plotjourneys(outside, AP, asa_buffered, samples =150)
plotjourneys(odinams, AP, asa_buffered, transit = True)
plotjourneys(odinams2, AP2, asa_buffered, transit = True)


def plot_modeshare(df):
# Convert the grouped data to a dictionary
    grouped = df.groupby(['aankpc', 'khvm']).size().unstack(fill_value=0)

    # Convert the grouped data to a dictionary
    result_dict = grouped.to_dict(orient='index')
    dict_df = pd.DataFrame.from_dict(result_dict, orient='index')

    # Set the index of the DataFrame to the larger set of 'aankpcs'
    larger_set = set(AP2.index)  # Larger set of 'aankpcs'
    dict_df = dict_df.reindex(larger_set, fill_value=0.01)
    merged_df = AP2.merge(dict_df, left_index=True, right_index=True, how='left')

    merged_df = merged_df.iloc[:10, :]
    total_cols = result_dict['1011'].keys()
    merged_df['proportions'] = merged_df[total_cols].div(merged_df[total_cols].sum(axis=1), axis=0).values.tolist()

    bounds = merged_df.geometry.bounds
    min_x = bounds['minx'].min()
    min_y = bounds['miny'].min()#52.225#
    max_x = bounds['maxx'].max()
    max_y = bounds['maxy'].max()#52.45

    fig = plt.figure()
    ax_map = fig.add_axes([0, 0, 1, 1])
    merged_df.plot(ax=ax_map, facecolor = 'None')
    ax_map.set_xlim([min_x, max_x])
    ax_map.set_ylim([min_y, max_y])

    for _, row in merged_df.iterrows():
        centroid = row['geometry'].centroid
        pie_data = row['proportions']
        lat, lon = centroid.x, centroid.y
        ax_bar = fig.add_axes([((lat - min_x +0.0001)/ (max_x-min_x))-.025, ((lon - min_y +0.0001)/ (max_y-min_y))-0.025, 0.05, 0.05])
        ax_bar.pie(pie_data, radius = 0.5)
        ax_bar.set_axis_off()
plot_modeshare(odinams2)


# def hypeodin():
# def hypeodin():
    # odin.merge(AP['centroid'], left_on = 'aankpc', right_on = AP.index)
    # odin['centroid'] = odin['a_centroid']
    # odin.merge(AP['centroid'], left_on = 'vertpc', right_on = AP.index)
    # odin['centroid'] = odin['v_centroid']
    # odin['v_centroid_within'] = odin['v_centroid'].apply(lambda centroid: asa.geometry.contains(centroid).any())
    # odin['a_centroid_within'] = odin['a_centroid'].apply(lambda centroid: asa.geometry.contains(centroid).any())


# odinams.merge(AP['centroid'], left_on ='aankpc', right_on = AP.index)#, suffixes = ('aankpc', 'aankpc'))
# # odinamsest = odinamsest.merge(AP['centroid'], left_on ='vertpc', right_on = AP.index, suffixes = ('aankpc', 'vertpc'))

# pc6=pd.read_csv('/Users/joshuathomas/Desktop/pc6_2022_v1.csv')
AP6 = gpd.read_file('PublicGeoJsons/pc6x.json').set_index('Postcode6')
pc6 = pd.read_pickle('PostcodeInfo/PC6Ams')

def Transit():
    fig, ax = plt.subplots()
    PCs.plot(ax = ax,facecolor = 'None', linewidth = 0.1)
    asa.plot(ax = ax, alpha = 0.7)

    tl['Marker_Size'] = tl['Modaliteit'].map({'Tram': 0.25, 'Metro': 3})
    tl['Color'] = tl['Modaliteit'].map({'Tram': 'y', 'Metro': 'r'})
    tl.plot(ax = ax, c = tl['Color'], markersize = tl['Marker_Size'], legend=True)
Transit()

def PCTOVERLAP(AP, asa):

    joined = gpd.sjoin(AP, asa.drop('index_right', axis =1), how='inner', op='intersects')

    overlapping_parts = gpd.GeoDataFrame(columns=joined.columns)

    # Iterate over the joined dataframe and find overlapping parts
    for index, row in joined.iterrows():
        overlapping_geometry = row['geometry']
        overlapping_part = asa.loc[asa.intersects(overlapping_geometry)]
        overlapping_parts = pd.concat([overlapping_parts, overlapping_part])

    # Merge overlapping parts based on geometry
    merged_parts = overlapping_parts.unary_union

    # Add overlapping parts as a new column in AP
    AP['Overlapping_Parts'] = AP.geometry.intersection(merged_parts)

    # Add the 'area_overlap' values to the 'AP' GeoDataFrame
    AP['area_overlap'] = AP['Overlapping_Parts'].area
    AP['area_overlap'] = AP['area_overlap'].fillna(0)
    AP['area'] = AP.geometry.area
    AP['percentage_overlap'] = 100*(AP['area_overlap'] /AP['area'])

    AP.plot(column = 'percentage_overlap', legend = True)

    return AP
AP = PCTOVERLAP(AP, asa)
def coverage(AP, threshold = 20):
    c = AP[AP.percentage_overlap > threshold]
    non_intersected_gdf = AP.loc[~AP.index.isin(c.index)]
    non_intersected_gdf.index = non_intersected_gdf.index.astype(str)
    fig, ax = plt.subplots()
    non_intersected_gdf.plot(ax=ax)
    PCs.plot(ax=ax, facecolor='None', linewidth=0.05)
    AP.plot(ax=ax, facecolor='None', linewidth=0.2)
    return non_intersected_gdf
non_intersected_gdf = coverage(AP)


# odin[odin.aankpc.isin(AP.Postcode4) & odin.vertpc.isin(AP.Postcode4)]
outside = odin[odin.aankpc.isin(non_intersected_gdf.index) & odin.vertpc.isin(non_intersected_gdf.index)]

outside.drop('CanFelyx', axis =1)
outside['CanFelyx'] = outside['oprijbewijsau'] | outside['oprijbewijsmo']
moti = outside[outside['khvm'] == 'Personenauto - bestuurder'].groupby(['khvm', 'doel', 'CanFelyx']).count()


info6 =AP6.join(pc6, how = 'inner')
intersected_gdf2 = sjoin(info6,asa.drop('index_right', axis =1), how="inner", op='intersects')
# non_intersected_gdf2 = AP6.loc[~AP6.index.isin(intersected_gdf2.index)]
fig, ax = plt.subplots()
intersected_gdf2.plot(ax = ax)
PCs.plot(ax = ax,facecolor = 'None', linewidth = 0.1)


desc = intersected_gdf2.describe().loc['mean']
comparison = pd.DataFrame({'SA': intersected_gdf2.describe().loc['mean'], 'NSA':info6.describe().loc['mean'],})
