from ModellingUtilities import *
from PlotUtilities import *
import pickle
from Optimise.PreOptimiser import get_num_target_areas

gdf = getAADO().to_crs(28992)
square_size = 150
with open("OptimiserResultNew", "rb") as fp:  # Unpickling
    res = pickle.load(fp)
grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
resulting_areas, f = res[0], res[1]
ASA_grid = get_num_target_areas(grid)

def plotstart_and_end():
    unchanged = list(set(resulting_areas).intersection(set(f)))
    fig, ax = plt.subplots()
    grid.to_crs(28992).loc[resulting_areas].plot(ax = ax, color = 'green')
    # grid.to_crs(28992).loc[unchanged].plot(ax = ax, color = 'yellow')
    # grid2grid_journeys_nonempty.to_crs(28992).plot(ax = ax, edgecolor = 'black')
    gdf.to_crs(28992).plot(ax = ax, facecolor = 'None')
    ax.set_title('end')
    fig, ax = plt.subplots()
    grid.to_crs(28992).loc[f].plot(ax = ax, color = 'red')
    grid.to_crs(28992).loc[unchanged].plot(ax = ax, color = 'yellow')
    # grid2grid_journeys_nonempty.to_crs(28992).plot(ax = ax, edgecolor = 'black')
    gdf.to_crs(28992).plot(ax = ax, facecolor = 'None')
    ax.set_title('start')

def viewresults(resulting_areas):
    fig, ax = plt.subplots()
    grid.to_crs(28992).loc[resulting_areas].plot(ax = ax, color = 'green')
    gdf.plot(ax = ax, facecolor = 'None')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

viewresults(resulting_areas)
viewresults(ASA_grid.index)

def remove_labels(axes):
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.set_xticks([])
    axes.set_yticks([])
    return

def comparetoasa():

    fig, [ax1, ax2] = plt.subplots(1, 2)
    grid.to_crs(28992).loc[resulting_areas].plot(ax = ax1, color = 'green')
    gdf.to_crs(28992).plot(ax = ax1, facecolor = 'None', linewidth = 0.02)
    GemPlot(ax1)
    ax1.set_title('sugg')
    remove_labels(ax1)

    ASA_grid.plot(ax=ax2)
    gdf.to_crs(28992).plot(ax=ax2, facecolor='None', linewidth = 0.02)
    GemPlot(ax2)
    ax2.set_title('real')
    remove_labels(ax2)

comparetoasa()

def GemPlot(axes):
    Gem = gpd.read_file('PublicGeoJsons/Gemeenten.geojson')
    staats = ['Diemen', 'Amstelveen', 'Ouder-Amstel', 'Amsterdam']
    Gem = Gem[Gem.statnaam.isin(staats)]
    Gem.boundary.plot(ax=axes, color='lightgrey', linewidth=0.5)
    return






fig, ax = plt.subplots()
ASA_grid.plot(ax = ax)
gdf.to_crs(28992).plot(ax = ax, facecolor = 'None')

fig, ax = plt.subplots()
grid.to_crs(28992).loc[fixed_areas].plot(ax = ax, color = 'green')
# grid.to_crs(28992).loc[unchanged].plot(ax = ax, color = 'yellow')
# grid2grid_journeys_nonempty.to_crs(28992).plot(ax = ax, edgecolor = 'black')
gdf.to_crs(28992).plot(ax = ax, facecolor = 'None')