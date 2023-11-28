from ModellingUtilities import *
from PlotUtilities import *
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from matplotlib.colors import Normalize

square_size = 100
city = 'Rotterdam'
cnndf = pd.read_pickle(os.path.join('CNN_Demand','CNN_Data',city, str(square_size))).fillna(0)#.drop('index_right', axis= 1)




def prepare(cnndf, city, target, channels = ['all_passengers', 'density', 'banned_area_proportion', 'counts_tram']):
    # channels = ['counts_train', 'density'] #'Man','density' 'Horeca', 'counts_metro'
    width, height = get_dimensions(cnndf, square_size)
    SA = pd.read_pickle('Misc/' + city + 'ServiceArea')
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
def get_modelling_data(rotations=[0, 90, 180, 270], TRAIN_RATIO = 0.8, BATCH_SIZE = 32):
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
def make_predictions():
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
def compare(prediction_df, val_only = False, transit = False, vmax = 2000):
    if val_only == True:
        # prediction_df = prediction_df.loc[m.reset_index().loc[val_loader.dataset.indices]['1D_index'].values]
        prediction_df = prediction_df[prediction_df['val'] ==True]
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
class SimpleDemandPredictor(nn.Module):
    def __init__(self, C):
        super(SimpleDemandPredictor, self).__init__()
        self.conv1 = nn.Conv2d(C, 16, kernel_size=window_size, padding=0)  # No padding needed here
        self.fc1 = nn.Linear(16, 6)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()
class DemandPredictor(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(DemandPredictor, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 max pooling

        # After pooling, 25x25 becomes 12x12
        self.fc = nn.Linear(int(hidden_channels * (window_size/2) * (window_size/2)), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class FDemandPredictor(nn.Module):
    def __init__(self, C):
        super(FDemandPredictor, self).__init__()

        self.conv1 = nn.Conv2d(C, 12, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)


        self.fc1 = nn.Linear(64 * (window_size // 2) * (window_size // 2), 6)
        self.fc1 = nn.Linear(64, 6)# Assuming one pooling operation
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        # x = x[:, :, 4, 4]
        middle_y = x.size(2) // 2
        middle_x = x.size(3) // 2
        x = x[:, :, middle_y, middle_x]
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.shape)
        return x.squeeze()
class ImprovedDemandPredictor(nn.Module):
    def __init__(self, C):
        super(ImprovedDemandPredictor, self).__init__()

        self.conv1 = nn.Conv2d(C, 12, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32, 16)  # Updated this line
        self.dropout = nn.Dropout(0.9)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):

        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)

        middle_y = x.size(2) // 2
        middle_x = x.size(3) // 2
        x = x[:, :, middle_y, middle_x]

        print(x.shape)

        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.shape)

        return x.squeeze()
class JDemandPredictor(nn.Module):
    def __init__(self, C):
        super(JDemandPredictor, self).__init__()

        self.conv1 = nn.Conv2d(C, 12, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(3, 3)

        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(24)

        # self.fc1 = nn.Linear(64 * (window_size // 2) * (window_size // 2), 6)
        self.fc1 = nn.Linear(24, 6)  # Assuming one pooling operation
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(6, 1)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        print(x.shape)

        # middle_y = x.size(2) // 2
        # middle_x = x.size(3) // 2
        # x = x[:, :, middle_y, middle_x]

        x = self.pool(x)

        print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)

        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()
class DemandPredictorgpt(nn.Module):
    def __init__(self, C, window_size=7):
        super(DemandPredictorgpt, self).__init__()

        self.conv1 = nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # We will dynamically set this later
        self.fc1_input_size = None

        self.fc1 = None
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        if self.fc1 is None:
            self.fc1_input_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(self.fc1_input_size, 128)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x.squeeze()
class FDemandPredictor_simple(nn.Module):
    def __init__(self, C, L2):
        super(FDemandPredictor_simple, self).__init__()

        self.conv1 = nn.Conv2d(C, 12, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)


        self.fc1 = nn.Linear(32 * (window_size // 2) * (window_size // 2), 6)
        self.fc1 = nn.Linear(32, L2)# Assuming one pooling operation
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(L2, 1)

    def forward(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        # x = x[:, :, 4, 4]
        middle_y = x.size(2) // 2
        middle_x = x.size(3) // 2
        x = x[:, :, middle_y, middle_x]
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.dropout(x)
        x = self.fc2(x)
        # print(x.shape)
        return x.squeeze()
class JeroenDemandPredictor(nn.Module):
    def __init__(self, C, C1, C2):
        super(JeroenDemandPredictor, self).__init__()

        self.C2 = C2

        self.conv1 = nn.Conv2d(C, C1, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(C1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(C1, C2, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(C2)

        self.fc1 = nn.Linear(C2 * 1 * 1, 3)

        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.conv2(x)))

        x = self.pool(x)

        x = x.view(-1, self.C2 * 1 * 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()



channels = ['sum_Passengers',
       'area_intersected_no_parking',
       'area_intersected_park',
       'aantal_inwoners', 'amenities', 'Horeca', 'university', 'tram', 'metro', 'area_intersected_water'] #_25_tot_45_jaar',
channels = ['amenities', 'area_intersected_no_parking']#, 'sum_Passengers', 'metro', 'aantal_inwoners']
window_size = 9





padded_channels, tensor_targets, width, height, C, ASA_grid = prepare(cnndf,city, 'counts_fstart_' + 'Morning' , channels)
train_loader, val_loader, prediction_data = get_modelling_data()

model = FDemandPredictor_simple(C, 3) #JeroenDemandPredictor(C, 8, 24)
shapes = False
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 4
for epoch in range(epochs):
    model.train()  # set model to training mode
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    # Validation
    model.eval()  # set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item() * inputs.size(0)

    val_loss = val_loss / len(val_loader.dataset)

    print(f"Ep [{epoch + 1}/{epochs}] - Train: {round(train_loss/1):.4f}, Validation: {round(val_loss/1):.4f}")

simple = True
def plot_preds(edgecolor = 'red', context = False, alpha =1, quantile = 0.95):
    prediction_df, all_predictions = make_predictions()
    vmax_ = prediction_df[prediction_df['ASA'] == 1].target.quantile(quantile)
    fig, ax = plt.subplots()
    remove_labels(ax)
    prediction_df.plot(alpha = alpha, column = 'pred', ax = ax,
                       vmin = 0, vmax = vmax_, legend = True)
    # gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax = ax, linewidth = 0.25)
    cnndf[cnndf['sum_Passengers']>0].plot(ax = ax, facecolor = 'None', edgecolor = edgecolor)
    cnndf[cnndf['metro']>0].plot(ax = ax, facecolor = 'None', edgecolor = edgecolor)
    if context == True:
        cx.add_basemap(ax, crs=cnndf.crs.to_string(), source=cx.providers.Stamen.Watercolor)
    else:
        getAADO(water = False, outline = False).to_crs(28992).plot(ax=ax, facecolor='None', linewidth=0.15)
    return prediction_df
prediction_df = plot_preds()


# cnndf[cnndf['banned_area_proportion']>0].plot(column = 'banned_area_proportion',ax = ax, alpha = 0.075, cmap = 'Greens')

prediction_df.to_pickle('CNN Demand/CNN_Predictions/Jdens5' + str(square_size))
compare(prediction_df, transit = False, vmax = prediction_df[prediction_df['ASA'] == 1].target.quantile(0.99))
compare(prediction_df, val_only= True, vmax = prediction_df[prediction_df['ASA'] == 1].target.quantile(0.95))
























fig, ax = plt.subplots()
# cnndf.plot(column = 'counts_fstart', ax = ax, cmap = 'Blues', vmax = cnndf.counts_fstart.max()/2, legend = True)
remove_labels(ax)
# getAADO().to_crs(28992).plot(ax = ax, facecolor = 'None', linewidth = 0.25)
cnndf.plot(column = 'young',ax = ax, cmap = 'Reds', legend = True)














fig, ax = plt.subplots()
# cnndf.plot(column = 'counts_fstart', ax = ax, cmap = 'Blues', vmax = cnndf.counts_fstart.max()/2, legend = True)
remove_labels(ax)
getAADO().to_crs(28992).plot(ax = ax, facecolor = 'None', linewidth = 0.25)
cnndf[cnndf['banned_area_proportion']>0].plot(column = 'banned_area_proportion',ax = ax, alpha = 0.05, cmap = 'Reds')


fig, ax = plt.subplots()
cnndf.plot(column = 'counts_park', cmap = 'Reds', ax = ax, alpha = 0.4)
# cnndf[cnndf.counts_Passengers > 1].plot(column = 'Horeca', ax = ax, cmap = 'Blues')
getAADO().to_crs(28992).plot(ax = ax, facecolor = 'None', linewidth = 0.25)

# fig, ax = plt.subplots()
# cnndf.plot(column = 'counts_park', cmap = 'Reds', ax = ax, alpha = 0.4)
# cnndf.plot(column = 'Horeca', ax = ax, cmap = 'Blues', vmax = cnndf.Horeca.max() / 3)
#
#
# fig, ax = plt.subplots()
# cnndf.plot(column = 'density', ax = ax, cmap = 'Blues', vmax = cnndf.density.max())
# getAADO().to_crs(28992).plot(ax = ax, facecolor = 'None', linewidth = 0.5)
# cnndf.plot(column = 'counts_park', cmap = 'Reds', ax = ax, alpha = 0.1)
#
#
# fig, ax = plt.subplots()
# cnndf.plot(column = 'counts_park', cmap = 'Reds', ax = ax, alpha = 0.4)
# cnndf[cnndf.counts_Passengers > 1].plot(column = 'counts_Passengers', ax = ax, cmap = 'Blues')
#
