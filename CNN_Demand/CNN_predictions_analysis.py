from ModellingUtilities import *
from matplotlib.colors import Normalize
from PlotUtilities import *
prediction_df_anal = pd.read_pickle('CNN Demand/CNN_Predictions/J150')
prediction_df_anal = pd.read_pickle('CNN Demand/CNN_Predictions/Jdens150')
# prediction_df.pred = prediction_df.pred.apply(lambda x: x[0])
# prediction_df.plot(column = 'pred')
# prediction_df.overlay(pd.read_pickle('PublicGeoJsons/AADO_PC4.geojson').to_crs(28992)).plot(column = 'pred')
# prediction_df.overlay(pd.read_pickle('PublicGeoJsons/AADO_PC4.geojson').to_crs(28992)).plot(column = 'actual', legend = 'True')

HP = gpd.read_file('PublicGeoJsons/HousingPrices.json')
HP.plot(column = 'SELECTIE')

def compare(prediction_df, val_only = True):
    if val_only == True:
        prediction_df = prediction_df.loc[val_loader.dataset.indices]

    vmax = prediction_df.pred.max()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    prediction_df[prediction_df['ASA'] == 1].plot(ax = ax1, column = 'pred', legend = False, vmax = vmax)
                                                  # norm=Normalize(vmin=0, vmax=1500), cmap = 'coolwarm')
    # trains.to_crs(28992).overlay(getAADO().to_crs(28992)).plot(ax=ax1, markersize = 3)
    # pd.read_pickle('Misc/AADOServiceArea').to_crs(28992).plot(ax = ax1, alpha = 0.7, color = 'green')
    # metro = gpd.read_file('PublicGeoJsons/TransitLines.json')
    # metro[metro.Modaliteit == 'Metro'].to_crs(28992).plot(ax=ax1, markersize = 3)
    # trains = pd.read_csv('PublicGeoJsons/Transport/stations-2022-01-nl.csv').set_index('code')
    # SPN = pd.read_csv('PublicGeoJsons/Transport/StationPassengerNumbers.csv').set_index('CodeUpper')
    # trains = trains.join(SPN[['Passengers']])
    # trains = gpd.GeoDataFrame(
    #     trains, geometry=gpd.points_from_xy(trains.geo_lng, trains.geo_lat), crs="EPSG:4326"
    # ).to_crs(28992)[['geometry', 'name_short', 'Passengers']]



    prediction_df[prediction_df['ASA'] == 1].plot(ax = ax2, column = 'target',  legend = False, vmax = vmax)
                                                  # norm=Normalize(vmin=0, vmax=1500), cmap = 'coolwarm')
    # gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax=ax1)
    ax1.set_title('Predicted')
    ax2.set_title('Actual')
    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
compare(prediction_df_anal, val_only = False)


fig,ax =plt.subplots()
prediction_df_anal.overlay(gdfA).plot(column = 'pred', cmap ='Reds', ax = ax,
                                      vmax =prediction_df_anal.pred.max()/2)
remove_labels(ax)
# merged_gdf.plot(facecolor = 'None', linewidth = 0.05, ax = ax, color = 'Black')
# cx.add_basemap(ax, crs=prediction_df_anal.crs.to_string(), alpha =0.2 )




def error(trans = False):

    prediction_df['error'] = prediction_df['pred'] - prediction_df['target']
    fig, ax = plt.subplots()
    prediction_df[prediction_df['ASA'] == 1].plot(column = 'error', ax = ax, legend = True,
                                                  norm=Normalize(vmin=-1000, vmax=1000), cmap = 'coolwarm')
    if trans == True:
        gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax = ax, linewidth = 0.5)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
error(trans = True)


def get_transfer_modelling_data():
    data_entries = []
    for i in range(height):
        for j in range(width):
            subgrid = padded_channels[:, i:i + window_size, j:j + window_size]

            # Convert subgrid tensor to numpy array
            if subgrid.is_cuda:
                subgrid = subgrid.cpu()
            subgrid_np = subgrid.numpy()

            # Calculate 1D index
            one_d_index = i * width + j

            # Convert tensor_targets to numpy array
            target_np = tensor_targets[i, j].item()

            # Iterate over the desired rotation angles and add to data_entries
            for angle in [0, 90]:#, 180, 270]:
                rotated_subgrid = rotate_subgrid(subgrid_np, angle)
                entry = {
                    'i_index': i,
                    'j_index': j,
                    '1D_index': one_d_index,
                    'rotation_angle': angle,
                    'subgrid': rotated_subgrid,
                    'target': target_np
                }
                data_entries.append(entry)
    all_subgrid_data = pd.DataFrame(data_entries)
    modelling_data = all_subgrid_data[ all_subgrid_data['1D_index'].isin(ASA_grid.index)]
    prediction_data = all_subgrid_data[all_subgrid_data['rotation_angle'] == 0]

    subgrids_tensor = torch.stack([torch.tensor(x) for x in modelling_data['subgrid'].values]).float()
    targets_tensor = torch.stack([torch.tensor(x) for x in modelling_data['target'].values]).float()

    return prediction_data, subgrids_tensor, targets_tensor
for city in ['Rotterdam', 'AADO']:
    cnndf = pd.read_pickle(os.path.join('CNN Demand', 'CNN_Data', city, str(square_size)))
    padded_channels, tensor_targets, width, height, C, ASA_grid = prepare(cnndf, city)
    prediction_data, subgrids_tensor, targets_tensor= get_transfer_modelling_data()
    if city == 'Rotterdam':
        train_loader = torch.utils.data.DataLoader(list(zip(subgrids_tensor, targets_tensor)), batch_size=32, shuffle=True)
    if city == 'AADO':
        val_loader = torch.utils.data.DataLoader(list(zip(subgrids_tensor, targets_tensor)), batch_size=32, shuffle=True)

def make_transfer_predictions():
    subgrids_tensor = torch.stack([torch.tensor(x) for x in prediction_data['subgrid'].values]).float()
    targets_tensor = torch.stack([torch.tensor(x) for x in prediction_data['target'].values]).float()
    full_loader = torch.utils.data.DataLoader(list(zip(subgrids_tensor, targets_tensor)), batch_size=32)
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for inputs, labels in full_loader:
            outputs = model(inputs.float())  # Ensure inputs are float32
            all_predictions.extend(outputs.numpy())
        if simple == False:
            all_predictions = [item for sublist in all_predictions for item in sublist]

    prediction_data['pred'] =  all_predictions
    prediction_df = cnndf[['geometry']].join(prediction_data.set_index('1D_index'))
    prediction_df['ASA'] = 0
    prediction_df['ASA'].iloc[ASA_grid.index] = 1
    return prediction_df, all_predictions
prediction_df, all_predictions = make_transfer_predictions()
prediction_df['pred_to_ams'] = prediction_df['pred'] / 2