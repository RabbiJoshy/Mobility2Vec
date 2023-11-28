import pandas as pd
import geopandas as gpd
import numpy as np
from ModellingUtilities import getAADO
import contextily as cx
from PlotUtilities import *


AP6 = gpd.read_file('PublicGeoJsons/pc6/pc6x.json').set_index('Postcode6')

Pc5 = gpd.read_file('/Users/joshuathomas/Desktop/2023-cbs_pc5_2022_v1/cbs_pc5_2022_v1.gpkg')
Pc5 = Pc5.replace(-99997, 0)

Pc5[Pc5.postcode5.str.startswith('1')].plot(column = 'aantal_inwoners', legend = True)

Pc5.sample(10000).plot(column = 'aantal_inwoners', legend = True)
townships = gpd.read_file('PublicGeoJsons/OtherCities/townships.geojson').to_crs(28992)
townships[townships.name == 'Amsterdam'].overlay(Pc5.to_crs(28992)).plot()


# AP6.plot()
square_size = 150
grid = pd.read_pickle('Demand Modelling/Grids/' + 'AADO' + '/' + str(square_size))

grid = grid.sjoin(AP6.to_crs(28992))
h = grid[['geometry','Aantal_ingebruik']].groupby('geometry').sum().reset_index()
h = gpd.GeoDataFrame(h, geometry = 'geometry', crs = 28992)
h['density_6'] = h['Aantal_ingebruik'] * ((1000/square_size)**2)/4
grid150 = pd.read_pickle('Demand Modelling/Grids/' + 'AADO' + '/' + str(square_size))
gridstep1 = grid150.merge(h[['density_6','geometry']], on = 'geometry', how = 'outer')
infopc4 = pd.read_pickle('PostcodeInfo/PC4_Clean_Geo')
infogrid = grid150.sjoin(infopc4)
i = infogrid[~infogrid.index.duplicated(keep='first')]
gridstep1 = gridstep1.join(i[['density']])
gridstep1['density_6'] = gridstep1['density_6'].fillna(gridstep1['density'])

gridstep1.plot(column = 'density_6')
gridstep1.to_pickle('density150')


fig, ax = plt.subplots()
# vmax = ii.density_1.max()
# infopc4.overlay(grid).plot(column = 'density', ax = ax, vmax = vmax, cmap = 'Reds')
gridstep1.plot(column = 'density_6', ax = ax, legend = True, cmap = 'Reds')
cx.add_basemap(ax, crs=grid150.crs.to_string())
remove_labels(ax)

#PC6 information turned out to be useless without accompanying shapefiles - com back to later?
def makepc6AADO():
    pc6=pd.read_csv('/Users/joshuathomas/Desktop/ThesisLargeFiles/pc6_2022_v1.csv')
    pc6.columns
    AADO = getAADO().index
    pc6['pc4'] = pc6['Postcode-6'].apply(lambda x: (x[:4]))
    pc6 = pc6[pc6.pc4.isin(AADO)]
    pc6.to_pickle('PublicGeoJsons/pc6AADO')
    return pc6

pc6 = makepc6AADO()

pc6 = pc6[pc6['Postcode-6'].isin(AP6.index.unique())].set_index('Postcode-6')
pc6 = pc6.replace('-99997', 0)
pc6 = pc6.replace('-99997.0', 0)
pc6 = pc6.replace(-99997, 0)
pc6[pc6.columns]  = pc6[pc6.columns].astype(float)
pc6[pc6.columns]  = pc6[pc6.columns].astype(int)

def replace_zeros(row, Total, cols, pct = False):
    zero_count = (row[cols] == 0).sum()   # Number of zeros in B and C
    non_zero_sum = row[cols][row[cols] != 0].sum()  # Sum of non-zero elements in B and C

    if non_zero_sum != 0:
        if non_zero_sum > 100:
            row[cols] *= (100/non_zero_sum)
        else:
        # Compute replacement value, and handle division by zero
            if pct == True:
                replace_value = (100 - non_zero_sum) / zero_count if zero_count != 0 else np.nan
            else:
                replace_value = (row[Total] - non_zero_sum) / zero_count if zero_count != 0 else np.nan

            # Replace zeros with the computed value
            row[cols] = row[cols].replace(0, replace_value)

    return row

# Apply the function to each row
df = pc6
df = df.apply(replace_zeros, args =  ('Totaal',['Man', 'Vrouw']), axis=1)
df = df.apply(replace_zeros, args =  ('Totaal',['tot 15 jaar', '15 tot 25 jaar','25 tot 45 jaar', '45 tot 65 jaar', '65 jaar en ouder']), axis=1)
df.columns
Geboren = ['Geboren in Nederland met een Nederlandse herkomst',
       'Geboren in Nederland met een Europese herkomst (excl. Nederland)',
       'Geboren in Nederland met herkomst buiten Europa',
       'Geboren buiten Nederland met een Europese herkomst (excl. Nederland)',
       'Geboren buiten Nederland met een herkomst buiten Europa']
df = df.apply(replace_zeros, args =  ('Totaal',Geboren, True), axis=1)
HH = [ 'Eenpersoons', 'Meerpersoons \nzonder kinderen', 'Eenouder',
       'Tweeouder', 'Huishoudgrootte']
df = df.apply(replace_zeros, args =  ('Totaal.1',HH), axis=1)
BA = ['voor 1945',
       '1945 tot 1965', '1965 tot 1975', '1975 tot 1985', '1985 tot 1995',
       '1995 tot 2005', '2005 tot 2015', '2015 en later']
df = df.apply(replace_zeros, args =  ('Totaal.2',BA), axis=1)
tt = ['Koopwoning', 'Huurwoning', 'Huurcoporatie', 'Niet bewoond']
df = df.apply(replace_zeros, args =  ('Totaal',tt, True), axis=1)

df.to_pickle('PostcodeInfo/PC6Ams')