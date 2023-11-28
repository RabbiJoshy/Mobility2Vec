from ModellingUtilities import *
from PlotUtilities import *
import random
from scipy.sparse import csr_matrix
from Optimise.PreOptimiser import *

square_size = 150
grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
prediction_df = pd.read_pickle('CNN_Demand/CNN_Predictions/2J' + str(square_size))
prediction_df = pd.read_pickle('CNN_Demand/CNN_Predictions/Transfer' + str(square_size))
fweighted_journey_df = get_fweighted_journey_df(square_size, prediction_df)
fweighted_journey_df = pd.read_pickle('plz')

poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).set_index('index_right')
city = 'AADO'
grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', city ,str(square_size))).to_crs(28992)
import geopandas as gpd
sindex = grid.sindex
polygon_grid_overlap = {}
for poly_index, poly_row in poly.iterrows():
    poly_geom = poly_row['geometry']

    # Initialize the inner dictionary for this polygon
    grid_overlap_dict = {}

    # Find approximate matches with bounding box (much quicker)
    possible_matches_index = list(sindex.intersection(poly_geom.bounds))
    possible_matches = grid.iloc[possible_matches_index]

    # Loop through possible matches to confirm
    for grid_index, grid_row in possible_matches.iterrows():
        grid_geom = grid_row['geometry']

        if poly_geom.intersects(grid_geom):
            intersected_area = poly_geom.intersection(grid_geom).area
            percentage_overlap = (intersected_area / grid_geom.area) * 100
            grid_overlap_dict[grid_index] = percentage_overlap

    # Add the inner dictionary to the main dictionary
    polygon_grid_overlap[poly_index] = grid_overlap_dict


fweighted_journey_df['total_from_cell'] = fweighted_journey_df['felyx_weighted_summed_dict'].apply(lambda x: sum(x.values()))
for col in ['pred', 'total_from_cell']:
    fig, ax = plt.subplots()
    vmax = fweighted_journey_df[col].quantile(0.95)
    fweighted_journey_df.plot(column = col, ax = ax, cmap = 'Reds',
                              legend = True, vmax = vmax)
    gdf = getAADO(outline = True).to_crs(28992)
    gdf.plot(ax = ax, facecolor = 'None', linewidth = 0.25)
    ax.set_title(col)
    # cx.add_basemap(ax, crs=fweighted_journey_df.crs.to_string())
    remove_labels(ax)
# prediction_df.plot(column = 'pred', legend = True, vmin =0, cmap = 'Reds')


fweighted_journey_df = pd.read_pickle('grid2grid')

def get_banned(grid, threshold = 0.7, FixedGemeenten = ['Ouder-Amstel', 'Amstelveen']):
    grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
    parks  = pd.read_pickle('BannedGrids/ParkBannedGrid' + str(square_size))
    water = pd.read_pickle('BannedGrids/WaterBannedGrid' + str(square_size))
    banned_locations = pd.read_pickle('BannedGrids/BannedGrid' + str(square_size))
    banned = pd.concat([banned_locations, water, parks])

    banned = banned_locations[['geometry']].join(banned.reset_index()[['area_intersected', 'index']].groupby('index').sum())

    banned = banned.rename(columns = {'area_intersected':'banned_area_intersected'})
    grid_new = grid.join(banned['banned_area_intersected'])


    banned_areas = list(grid_new[grid_new['banned_area_intersected'] > threshold].index)

    Gemeenten = gpd.read_file('PublicGeoJsons/OtherCities/townships.geojson')
    Gemeenten = Gemeenten[Gemeenten.name.isin(FixedGemeenten)].to_crs(28992)
    FG = grid.sjoin(Gemeenten).index

    print(banned_areas)
    banned_areas = list(set(banned_areas).union(set(list(FG))))
    grid_new.loc[banned_areas].plot(column='banned_area_intersected')

    return grid_new, banned_areas, FG
def journey_dictionary_to_csr(journeys, max_area_id):
    data, row_indices, col_indices = [], [], []

    for from_area, destinations in journeys.items():
        for to_area, count in destinations.items():
            data.append(count)
            row_indices.append(int(from_area))
            col_indices.append(int(to_area))

    return csr_matrix((data, (row_indices, col_indices)), shape=(max_area_id + 1, max_area_id + 1))
def total_journeys_for_subset_csr(subset, matrix):
    # Extract rows for the subset
    submatrix_rows = matrix[subset, :]

    # Extract desired columns from the extracted rows
    submatrix = submatrix_rows[:, subset]

    # Return the sum of the submatrix after removing the diagonal
    return submatrix.sum() - submatrix.diagonal().sum()
def refine_subset2(subset, csr_matrix_, fixed_areas, banned_areas, num_iterations=1000):
    all_areas = set(jdict.keys())
    remaining_areas = all_areas - set(subset) - set(fixed_areas) - set(banned_areas)

    first_subset = subset

    best_journey_count = total_journeys_for_subset_csr(subset + fixed_areas, csr_matrix_)
    for it in range(num_iterations):
        if it % 1000 == 0:
            print(it)
            print(round(best_journey_count, -2))
        area_to_remove = random.choice(subset)
        area_to_add = random.choice(list(remaining_areas))

        new_subset = subset.copy()
        new_subset.remove(area_to_remove)
        new_subset.append(area_to_add)

        new_journey_count = total_journeys_for_subset_csr(new_subset + fixed_areas, csr_matrix_)
        if new_journey_count > best_journey_count:
            best_journey_count = new_journey_count
            subset = new_subset
            remaining_areas.remove(area_to_add)
            remaining_areas.add(area_to_remove)

    return subset
def random_sampling_with_refinement2(csr_matrix_, jdict, fixed_areas, banned_areas, target_areas, num_samples=50, num_iterations=1000, start_set = None):
    non_fixed_areas = list(set(jdict.keys()) - set(fixed_areas) - set(banned_areas))

    best_subset = []
    best_journey_count = 0

    for i in range(num_samples):
        import time

        random.seed(time.time())

        print(i)
        if start_set == None:
            print(len(non_fixed_areas), target_areas, len(fixed_areas))
            random_subset = random.sample(non_fixed_areas, target_areas - len(fixed_areas))
        else:
            random_subset = [x for x in start_set if x not in fixed_areas]

        if i == 0:
            first = random_subset

            # visualise(first)

        refined_subset = refine_subset2(random_subset, csr_matrix_, fixed_areas, banned_areas, num_iterations)

        # fig, ax = plt.subplots()
        # grid.to_crs(28992).loc[refined_subset + fixed_areas].plot(ax=ax)
        # gdf.to_crs(28992).plot(ax=ax, facecolor='None')
        # ax.set_title(str(i))
        # plt.show()

        refined_subset_journey_count = total_journeys_for_subset_csr(refined_subset + fixed_areas, csr_matrix_)

        if refined_subset_journey_count > best_journey_count:
            print('improved')
            best_journey_count = refined_subset_journey_count
            best_subset = refined_subset

    return fixed_areas + best_subset, first
def visualise(resulting_areas):
    fig, ax = plt.subplots()
    # grid.to_crs(28992).loc[f].plot(ax=ax)
    grid.to_crs(28992).loc[resulting_areas].plot(ax=ax, facecolor = 'green')
    gdf = getAADO(outline=True).to_crs(28992)
    gdf.to_crs(28992).plot(ax=ax, facecolor='None', linewidth = 0.15)
    remove_labels(ax)
    grid.to_crs(28992).loc[banned_areas].plot(ax = ax, color = 'red', alpha =0.2)
    grid.to_crs(28992).loc[GidxSA].plot(ax=ax, color='green')
    grid.to_crs(28992).loc[fixed_areas].plot(ax = ax, facecolor = 'None', edgecolor = 'yellow', linewidth = 1)
    cx.add_basemap(ax, crs=prediction_df.crs.to_string(), alpha =0.5)
    return

# jdict[10500]
#
# fig, ax = plt.subplots()
# gdf = getAADO(outline=True).to_crs(28992)
# gdf.plot(ax=ax, facecolor='None', linewidth=0.05)
# prediction_df.loc[[10500]].plot(ax = ax)
#
#
# fig, ax = plt.subplots()
# gdf = getAADO(outline=True).to_crs(28992)
# gdf.plot(ax=ax, facecolor='None', linewidth=0.05)
# for i in jdict[10500].keys():
#     print(i)
#     # prediction_df.loc[[i]].plot(ax = ax)
#     poly[poly.index_right == i].plot(ax = ax)
#
# poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992)
#


grid, banned_areas, Gidx = get_banned(grid, 0.5, [])# ['Ouder-Amstel', 'Amstelveen', 'Diemen'])
jdict = dict(zip(fweighted_journey_df.index, fweighted_journey_df['felyx_weighted_summed_dict']))
fixed_areas = transit_grid_squares = get_fixed_squares(jdict,150,  Metro = True)
target_areas = get_num_target_areas(grid)
GidxSA = [x for x in target_areas.index if x in Gidx]
num_target_areas = len(target_areas)
prestart = list(set(target_areas.index).difference(set(banned_areas)))
J_csr_matrix_ = journey_dictionary_to_csr(jdict, len(grid))

resulting_areas, f = random_sampling_with_refinement2(J_csr_matrix_, jdict, fixed_areas, banned_areas, num_target_areas,
                                                      start_set = prestart ,
                                                    num_samples= 1, num_iterations=3000)
visualise(resulting_areas)
resulting_areas, f = random_sampling_with_refinement2(J_csr_matrix_, jdict, fixed_areas, banned_areas, num_target_areas,
                                                      start_set = resulting_areas, num_samples=1, num_iterations= 30000)
visualise(resulting_areas) #56k highest so far

fig, ax = plt.subplots()
prediction_df.loc[jdict.keys()].plot(ax = ax)
cx.add_basemap(ax, crs=fweighted_journey_df.crs.to_string())





fig, ax = plt.subplots()
prediction_df[prediction_df.pred>200].plot(ax=ax, column = 'pred', cmap = 'Blues', vmax = prediction_df.pred.max())
gdf.to_crs(28992).plot(ax=ax, facecolor='None', linewidth =0.25)
cx.add_basemap(ax, crs=prediction_df.crs.to_string(), alpha =0.8)


fweighted_journey_df['sum_all'] = fweighted_journey_df.felyx_weighted_summed_dict.apply(lambda x:sum(x.values()))
fig, ax = plt.subplots()
gdf.to_crs(28992).plot(ax=ax, facecolor='None', linewidth =0.25)
fweighted_journey_df.plot(column = 'sum_all', cmap = 'Reds', ax = ax, alpha =0.8,
                          vmax = fweighted_journey_df.sum_all.max() /4)
cx.add_basemap(ax, crs=fweighted_journey_df.crs.to_string(), alpha =0.5)
remove_labels(ax)



import pickle
Result = [resulting_areas, f]
with open("OptimiserResult", "wb") as fp:
    pickle.dump(Result, fp)







