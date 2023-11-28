import matplotlib.pyplot as plt

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


fp_weighted_Grid2Centroid = pd.read_pickle('plz')

def polylgon_grid_overlap_():
    poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).set_index('index_right')
    city = 'AADO'
    grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', city ,str(square_size))).to_crs(28992)
    import geopandas as gpd
    sindex = grid.sindex

    # Initialize an empty dictionary to hold the results
    polygon_grid_overlap = {}

    # Loop through each polygon
    for poly_index, poly_row in poly.iterrows():
        poly_geom = poly_row['geometry']
        poly_area = poly_geom.area

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
                percentage_overlap = intersected_area / poly_area

                grid_overlap_dict[grid_index] = percentage_overlap

        # Add the inner dictionary to the main dictionary
        polygon_grid_overlap[poly_index] = grid_overlap_dict

    return polygon_grid_overlap
polygon_grid_overlap = polylgon_grid_overlap_()
# polygon_grid_overlap.to_pickle('')

def calculate_grid_to_grid_journeys(row):
    journeys = {}
    for polygon, count in row['felyx_weighted_summed_dict'].items():
        for grid, weight in polygon_grid_overlap.get(polygon, {}).items():
            journeys[grid] = journeys.get(grid, 0) + count * weight
    return journeys

# Use the apply function to perform the operation row-wise, storing the result as a new column
fp_weighted_Grid2Centroid['grid_to_grid_journeys'] = fp_weighted_Grid2Centroid.apply(calculate_grid_to_grid_journeys, axis=1)
fp_weighted_Grid2Centroid.to_pickle('grid2grid')
grid2grid = pd.read_pickle('grid2grid')[['geometry', 'grid_to_grid_journeys']]
grid2grid.to_pickle('grid2grid')



grid2grid = pd.read_pickle('grid2grid')
grid2grid['sum'] = grid2grid['grid_to_grid_journeys'].apply(lambda x: sum(x.values()))

grid2grid.plot(column = 'sum')






