from ModellingUtilities import *
# Params
square_size = 150
city = 'AADO'

# grid = pd.read_pickle(os.path.join('Demand Modelling', 'Grids', city, str(square_size))).to_crs(28992)
# poly = pd.read_pickle('Misc/AADO_VMA_polygons').to_crs(28992).set_index('index_right')
# spatial_index = grid.sindex

def calc_grid_polygon_overlap(grid, poly):
    result_df = grid[['geometry']].copy()
    result_df['overlap_dict'] = [{} for _ in range(len(result_df))]

    spatial_index = poly.sindex

    for idx, grid_row in grid.iterrows():
        if idx % 1000 == 0:
            print(f"Processing index: {idx}")

        possible_matches_index = list(spatial_index.intersection(grid_row['geometry'].bounds))
        possible_matches = poly.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(grid_row['geometry'])]

        for poly_idx, poly_row in precise_matches.iterrows():
            intersection_area = grid_row['geometry'].intersection(poly_row['geometry']).area
            percentage = (intersection_area / grid_row['geometry'].area) * 100
            result_df.at[idx, 'overlap_dict'][poly_idx] = percentage

    return result_df
# Execute the function
# overlap_df = calc_grid_polygon_overlap(grid, poly)
# overlap_df.to_pickle('overlap_df')


overlap_df = pd.read_pickle('overlap_df')
connections = pd.read_pickle('VMA_Connections' + str(square_size))
connections.CENTROIDNR_x

def weight_journeys_by_overlap(overlap_df, connections_df):
    weighted_df = overlap_df.copy()
    weighted_df['weighted_journeys'] = [{} for _ in range(len(weighted_df))]

    for idx, row in overlap_df.iterrows():
        grid_overlap_dict = row['overlap_dict']  # Contains the centroid composition of the grid cell

        weighted_journeys_dict = {}

        for centroid, overlap_weight in grid_overlap_dict.items():
            connections_row = connections_df[connections_df['CENTROIDNR_x'] == str(centroid)]

            if not connections_row.empty:
                connection_dict = connections_row.iloc[0]['connections_car_weighted']

                for destination, journey_percentage in connection_dict.items():
                    weighted_journey_percentage = overlap_weight * journey_percentage / 100
                    existing_value = weighted_journeys_dict.get(destination, 0)
                    weighted_journeys_dict[destination] = existing_value + weighted_journey_percentage

        weighted_df.at[idx, 'weighted_journeys'] = weighted_journeys_dict

    return weighted_df

def weight_journeys_by_overlap(overlap_df, connections_df):
    weighted_df = overlap_df.copy()
    weighted_df['weighted_journeys'] = [{} for _ in range(len(weighted_df))]

    for idx, row in overlap_df.iterrows():
        grid_overlap_dict = row['overlap_dict']
        weighted_journeys_dict = {}

        for centroid, overlap_weight in grid_overlap_dict.items():
            overlap_weight = 0 if pd.isna(overlap_weight) else overlap_weight
            connections_row = connections_df[connections_df['CENTROIDNR_x'] == str(centroid)]

            if not connections_row.empty:
                connection_dict = connections_row.iloc[0]['connections_car_weighted']

                for destination, journey_percentage in connection_dict.items():
                    journey_percentage = 0 if pd.isna(journey_percentage) else journey_percentage
                    weighted_journey_percentage = overlap_weight * journey_percentage / 100
                    existing_value = weighted_journeys_dict.get(destination, 0)
                    weighted_journeys_dict[destination] = existing_value + weighted_journey_percentage

        weighted_df.at[idx, 'weighted_journeys'] = weighted_journeys_dict

    return weighted_df

weighted_df = weight_journeys_by_overlap(overlap_df, connections)
weighted_df['wjsum'] = weighted_df['weighted_journeys'].apply(lambda x: sum([float(j) for j in x.values()]))
weighted_df.plot(column = 'wjsum')


prediction_df = pd.read_pickle('CNN_Demand/CNN_Predictions/J' +str(square_size))


# Join weighted_df and prediction_df on their indices
combined_df = weighted_df.join(prediction_df['pred'])

# Create the new column by multiplying the weighted_journeys with Count
combined_df['weighted_count_journeys'] = combined_df.apply(
    lambda row: {key: value * row['pred'] for key, value in row['weighted_journeys'].items()},
    axis=1
)


combined_df['wjsump'] = combined_df['wjsum'] * combined_df['pred']
combined_df.plot(column = 'wjsump')
combined_df.plot(column = 'pred')



combined_df.to_pickle('plz')
combined_df = combined_df.rename(columns = {'weighted_count_journeys' :'felyx_weighted_summed_dict'})




combined_df['see'] = combined_df['felyx_weighted_summed_dict'].apply(lambda x: sum([float(j) for j in x.values()]))
combined_df.plot(column = 'see')