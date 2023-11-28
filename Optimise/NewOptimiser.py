from ModellingUtilities import *
from PlotUtilities import *
import random
from scipy.sparse import csr_matrix
from Optimise.PreOptimiser import *

square_size = 150
grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
prediction_df = pd.read_pickle('CNN_Demand/CNN_Predictions/Transfer' + str(square_size))
fweighted_journey_df = pd.read_pickle('grid2grid')

def visualise_one_cell(grid2grid, start):
    fig, ax = plt.subplots()
    where = grid2grid.loc[start]['grid_to_grid_journeys']
    op = where.keys()
    opa = grid.loc[op]
    opa['count'] = where.values()

    opa.plot(column='count', ax = ax)

    gdf = getAADO(outline=True).to_crs(28992)
    gdf.to_crs(28992).plot(ax=ax, facecolor='None', linewidth = 0.15)
    remove_labels(ax)
    grid2grid.loc[[start]].plot(ax = ax, edgecolor = 'red', linewidth = 1.5)

    return
visualise_one_cell(fweighted_journey_df, 11469)


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
            print(round(best_journey_count/1000, -2))
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
def visualise(resulting_areas, info = True):
    fig, ax = plt.subplots()
    # grid.to_crs(28992).loc[f].plot(ax=ax)
    grid.to_crs(28992).loc[resulting_areas].plot(ax=ax, facecolor = 'green')
    gdf = getAADO(outline=True).to_crs(28992)
    gdf.to_crs(28992).plot(ax=ax, facecolor='None', linewidth = 0.15)
    remove_labels(ax)
    if info == True:
        grid.to_crs(28992).loc[banned_areas].plot(ax = ax, color = 'red', alpha =0.2)
        grid.to_crs(28992).loc[GidxSA].plot(ax=ax, color='green')
        grid.to_crs(28992).loc[fixed_areas].plot(ax = ax, facecolor = 'None', edgecolor = 'yellow', linewidth = 1)
        cx.add_basemap(ax, crs=prediction_df.crs.to_string(), alpha =0.5)
    return

total_journeys_for_subset_csr([11469, 11460], J_csr_matrix_)
ran = random.sample(list(grid.index), len(prestart))

grid, banned_areas, Gidx = get_banned(grid, 0.5, [])# ['Ouder-Amstel', 'Amstelveen', 'Diemen'])
jdict = dict(zip(fweighted_journey_df.index, fweighted_journey_df['grid_to_grid_journeys']))
fixed_areas = transit_grid_squares = get_fixed_squares(jdict,150,  Metro = True)
target_areas = get_num_target_areas(grid)
GidxSA = [x for x in target_areas.index if x in Gidx]
num_target_areas = len(target_areas)
prestart = list(set(target_areas.index).difference(set(banned_areas)))
J_csr_matrix_ = journey_dictionary_to_csr(jdict, len(grid))

resulting_areas, f = random_sampling_with_refinement2(J_csr_matrix_, jdict, fixed_areas, banned_areas, num_target_areas,
                                                      start_set = ran ,
                                                    num_samples= 1, num_iterations=3000)
visualise(resulting_areas, info = False)
resulting_areas, f = random_sampling_with_refinement2(J_csr_matrix_, jdict, fixed_areas, banned_areas, num_target_areas,
                                                      start_set = resulting_areas, num_samples=1, num_iterations= 10000)
visualise(resulting_areas)
visualise(prestart)


import pickle
Result = [resulting_areas, f]
with open("OptimiserResultNew", "wb") as fp:
    pickle.dump(Result, fp)
