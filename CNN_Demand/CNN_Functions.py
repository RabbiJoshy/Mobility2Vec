from ModellingUtilities import *
from PlotUtilities import *
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from matplotlib.colors import Normalize

def prepare(cnndf, city, target, channels, window_size, square_size = 150):

    width, height = get_dimensions(cnndf, square_size)
    ext_dict = {'AADO': 'Ams/2023-03-01', 'Rotterdam': 'Rott/Rotterdam --- groot service area',
                'Den Haag':'Den Haag/Den Haag --- 2022'} #'AADO': 'Ams/Amsterdam --- Update23032022'
    ext = ext_dict[city]
    SA = pd.read_pickle('Service Areas/' + ext)

    # SA = pd.read_pickle('Misc/' + city + 'ServiceArea')
    ASA_grid = cnndf[['geometry']].sjoin(SA.to_crs(28992)[['geometry']])
    df = cnndf[channels + [target]]
    df[channels] = (df[channels] - df[channels].min()) / (df[channels].max() - df[channels].min())
    df = df.rename(
        columns = dict(zip(channels, [f'channel{i}' for i in range(1, len(channels)+1)]))
    )
    df = df.rename(columns = {target: 'target'})

    C = len(channels)

    channels = [df[f'channel{i}'].values.reshape(height, width) for i in range(1, C+1)]
    tensor_channels = torch.stack([torch.tensor(channel) for channel in channels])
    targets_2d = df['target'].values.reshape(height, width)
    print(targets_2d)
    tensor_targets = torch.tensor(targets_2d)
    padded_channels = F.pad(tensor_channels, (window_size//2, window_size//2, window_size//2, window_size//2), mode='constant', value=0)

    return padded_channels, tensor_targets, width, height, C, ASA_grid
def rotate_subgrid(np_array, degrees):
    """Rotate the subgrid by the specified degrees."""
    if degrees == 90:
        return np.rot90(np_array, 1, axes=(-2, -1)).copy()
    elif degrees == 180:
        return np.rot90(np_array, 2, axes=(-2, -1)).copy()
    elif degrees == 270:
        return np.rot90(np_array, 3, axes=(-2, -1)).copy()
    return np_array  # If 0 degrees
# def get_modelling_data(ASA_grid, height, width, padded_channels, tensor_targets, window_size, rotations = [0, 90, 180, 270]):
#     data_entries = []
#     for i in range(height):
#         for j in range(width):
#             subgrid = padded_channels[:, i:i + window_size, j:j + window_size]
#
#             # Convert subgrid tensor to numpy array
#             if subgrid.is_cuda:
#                 subgrid = subgrid.cpu()
#             subgrid_np = subgrid.numpy()
#
#             # Calculate 1D index
#             one_d_index = i * width + j
#
#             # Convert tensor_targets to numpy array
#             target_np = tensor_targets[i, j].item()
#
#             # Iterate over the desired rotation angles and add to data_entries
#             for angle in rotations:#[0, 90, 180, 270]:
#                 rotated_subgrid = rotate_subgrid(subgrid_np, angle)
#                 entry = {
#                     'i_index': i,
#                     'j_index': j,
#                     '1D_index': one_d_index,
#                     'rotation_angle': angle,
#                     'subgrid': rotated_subgrid,
#                     'target': target_np
#                 }
#                 data_entries.append(entry)
#     all_subgrid_data = pd.DataFrame(data_entries)
#     modelling_data = all_subgrid_data[ all_subgrid_data['1D_index'].isin(ASA_grid.index)]
#     prediction_data = all_subgrid_data[all_subgrid_data['rotation_angle'] == 0]
#
#     subgrids_tensor = torch.stack([torch.tensor(x) for x in modelling_data['subgrid'].values]).float()
#     targets_tensor = torch.stack([torch.tensor(x) for x in modelling_data['target'].values]).float()
#
#     train_size = int(0.8 * len(subgrids_tensor))
#     val_size = len(subgrids_tensor) - train_size
#     train_data, val_data = torch.utils.data.random_split(list(zip(subgrids_tensor, targets_tensor)),
#                                                          [train_size, val_size], generator = torch.Generator().manual_seed(69))
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
#
#     return train_loader, val_loader, prediction_data, modelling_data
def make_predictions(cnndf, prediction_data, model, ASA_grid, simple = True):
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

    prediction_data['pred'] = all_predictions
    prediction_df = cnndf[['geometry']].join(prediction_data.set_index('1D_index'))
    prediction_df['ASA'] = 0
    prediction_df['ASA'].iloc[ASA_grid.index] = 1
    return prediction_df
def get_modelling_data(ASA_grid, height, width, padded_channels, tensor_targets, window_size, rotations=[0, 90, 180, 270], TRAIN_RATIO = 0.8, BATCH_SIZE = 32):
    data_entries = []

    for i in range(height):
        for j in range(width):
            subgrid = padded_channels[:, i:i + window_size, j:j + window_size]

            if subgrid.is_cuda:
                subgrid = subgrid.cpu()

            subgrid_np = subgrid.numpy()
            one_d_index = i * width + j
            target_np = tensor_targets[i, j].item()

            entry = {
                'i_index': i,
                'j_index': j,
                '1D_index': one_d_index,
                'rotation_angle': 0,
                'subgrid': subgrid_np,
                'target': target_np,
                'is_in_ASA': one_d_index in ASA_grid.index,
                'val': False
            }

            data_entries.append(entry)

    all_subgrid_data = pd.DataFrame(data_entries)

    modelling_indices = all_subgrid_data[all_subgrid_data['is_in_ASA']].index.tolist()
    np.random.shuffle(modelling_indices)

    train_size = int(TRAIN_RATIO * len(modelling_indices))
    train_indices = modelling_indices[:train_size]
    val_indices = modelling_indices[train_size:]

    all_subgrid_data.loc[train_indices, 'val'] = False
    all_subgrid_data.loc[val_indices, 'val'] = True

    train_data_df = all_subgrid_data.loc[modelling_indices].reset_index(drop=True)
    val_data_df = all_subgrid_data.loc[val_indices].reset_index(drop=True)

    augmented_train_data = []
    for _, row in train_data_df.iterrows():
        for angle in rotations:
            rotated_subgrid = rotate_subgrid(row['subgrid'], angle)
            augmented_entry = row.copy()
            augmented_entry['rotation_angle'] = angle
            augmented_entry['subgrid'] = rotated_subgrid
            augmented_train_data.append(augmented_entry)

    train_data_df = pd.DataFrame(augmented_train_data)

    train_subgrids_tensor = torch.stack([torch.tensor(x) for x in train_data_df['subgrid'].values]).float()
    train_targets_tensor = torch.stack([torch.tensor(x) for x in train_data_df['target'].values]).float()

    val_subgrids_tensor = torch.stack([torch.tensor(x) for x in val_data_df['subgrid'].values]).float()
    val_targets_tensor = torch.stack([torch.tensor(x) for x in val_data_df['target'].values]).float()

    train_loader = torch.utils.data.DataLoader(list(zip(train_subgrids_tensor, train_targets_tensor)),
                                               batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(list(zip(val_subgrids_tensor, val_targets_tensor)), batch_size=BATCH_SIZE)

    return train_loader, val_loader, all_subgrid_data
def compare(prediction_df, val_loader, val_only = False, transit = False, vmax = 2000):
    if val_only == True:
        prediction_df = prediction_df.loc[m.reset_index().loc[val_loader.dataset.indices]['1D_index'].values]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    prediction_df[prediction_df['ASA'] == 1].plot(ax = ax1, column = 'pred', legend = False,
                                                  norm=Normalize(vmin=0, vmax=vmax), cmap = 'magma')
    prediction_df[prediction_df['ASA'] == 1].plot(ax = ax2, column = 'target',  legend = False,
                                                  norm=Normalize(vmin=0, vmax=vmax), cmap = 'magma')
    if transit == True:
        gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax=ax1, linewidth=0.25)
        gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax=ax2, linewidth = 0.25)
    ax1.set_title('Predicted')
    ax2.set_title('Actual')
    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    return