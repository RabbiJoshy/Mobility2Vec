import pandas as pd
import geopandas as gpd
Extrainfo = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021All')
g = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')

rp = g.representative_point()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
rp.plot(ax = ax)
g.plot(ax = ax, facecolor = 'None')

ff = pd.read_pickle('FakeFelyx/FakeFelyxData')
fk= ff[ff.Real == 0]
fk.aankpc