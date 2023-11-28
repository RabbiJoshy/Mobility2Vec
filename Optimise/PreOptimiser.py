from ModellingUtilities import *

square_size = 150
# grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
# prediction_df = pd.read_pickle('CNN_Demand/CNN_Predictions/J' +str(square_size))

# def banned_squares():
#     return
def get_grid2grid_journeys(square_size, better = True):
    if better == True:
        grid2grid_journeys = pd.read_pickle('Demand Modelling/Grid2Grid/bettergrid2grid_journeys_' + str(square_size))
    else:
        grid2grid_journeys = pd.read_pickle('Demand Modelling/Grid2Grid/bettergrid2grid_journeys_' + str(square_size))
    grid2grid_journeys_nonempty = grid2grid_journeys.loc[grid2grid_journeys['summed_dict'].str.len() > 0]
    return grid2grid_journeys_nonempty
def get_fixed_squares(data, square_size, Metro = False):
    grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
    transit = gpd.read_file('PublicGeoJsons/transport/TransitLines.json')
    if Metro == True:
        transit = transit[transit['Modaliteit'] == 'Metro']
    transit_grid = grid.to_crs(28992).sjoin(transit.to_crs(28992), how = 'inner')
    transit_grid_squares = [x for x in list(transit_grid.index) if x in data.keys()]
    return transit_grid_squares
def get_num_target_areas(grid):
    #The alternative would be to get the actual total area of the FSA and not just the gridded number
    # grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', 'AADO', str(square_size))).to_crs(28992)
    #
    ASA = getASA()[['geometry']]
    ASA_grid = grid.to_crs(28992).sjoin(ASA.to_crs(28992), how='inner')
    # ASA_grid_squares = [x for x in list(ASA_grid.index) if x in data.keys()]
    return ASA_grid
def get_fweighted_journey_df(square_size, prediction_df):
    jdict = get_grid2grid_journeys(square_size)
    jdict = jdict.join(prediction_df['pred'])
    jdict['felyx_weighted_summed_dict'] = [
        {k: v * pred for k, v in d.items()} for pred, d in zip(jdict['pred'], jdict['summed_dict'])
    ]
    return jdict

# fweighted_journey_df = get_fweighted_journey_df(square_size, prediction_df)
