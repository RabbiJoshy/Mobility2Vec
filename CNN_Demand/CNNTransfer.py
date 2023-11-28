from CNN_Demand.CNN_Functions import *
import torch.nn.functional as F

square_size = 100
window_size = 9
testcity = 'AADO'
traincity = 'Den Haag'
cnndf_train = pd.read_pickle(os.path.join('CNN_Demand','CNN_Data',traincity, str(square_size))).fillna(0)
cnndf_test = pd.read_pickle(os.path.join('CNN_Demand','CNN_Data', testcity, str(square_size))).fillna(0)


channels = [ 'metro',
       'amenities', 'Horeca','area_intersected_park','area_intersected_water',
             'aantal_inwoners_25_tot_45_jaar','university',  'tram', 'sum_Passengers', 'area_intersected_no_parking']


# for i in ['counts_fstart_' + x for x in ['Other', 'Morning', 'Evening']]:
#     city = testcity
#     fig, ax = plt.subplots()
#     cnndf_test[cnndf_test[i]>0].plot(column = i, ax = ax,  cmap = 'Reds')
#     # pd.read_pickle('PublicGeoJsons/OtherCities/' + city + 'PC4.geojson').to_crs(28992).plot(ax=ax, facecolor='None',
#     #                                                                                         linewidth=0.15)
#     # getAADO(water=False, outline = True).to_crs(28992).plot(ax=ax, facecolor='None', linewidth=0.15)
#     # cx.add_basemap(ax, crs=cnndf.crs.to_string())
#     ax.set_title(i)
#     remove_labels(ax)
# # cnndf_test[cnndf_test.counts_fstart >0].plot(column = 'counts_fstart', legend = True)

def compare_transfer(prediction_df, transit=False, pred_column = 'pred'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    vmax_value = prediction_df[pred_column].quantile(0.99)

    # Create plots
    im1 = prediction_df[prediction_df['ASA'] == 1].plot(
        ax=ax1, column= pred_column, legend=False,
        vmax=vmax_value, cmap='Reds')
    im2 = prediction_df[prediction_df['ASA'] == 1].plot(
        ax=ax2, column='target', legend=False,
        vmax=vmax_value, cmap='Reds')

    # Add title
    ax1.set_title('Predicted')
    ax2.set_title('Actual')

    # Add basemap
    cx.add_basemap(ax1, crs=prediction_df.crs.to_string(), source=cx.providers.CartoDB.Voyager)
    cx.add_basemap(ax2, crs=prediction_df.crs.to_string(), source=cx.providers.CartoDB.Voyager)

    # Hide axes ticks
    for ax in [ax1, ax2]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # # Create an axes for colorbar
    # cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    #
    # # Create colorbar
    # norm = Normalize(vmin= prediction_df[pred_column].min(), vmax=vmax_value)
    # sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    # sm.set_array([])
    # fig.colorbar(sm, cax=cax, orientation='vertical')
    #
    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
    # plt.show()
    return

class FDemandPredictor(nn.Module):
    def __init__(self, C, window_size):
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
        x = F.relu(self.bn1(self.conv1(x)))
        # torch.Size([32, 12, 5, 5])
        x = F.relu(self.bn2(self.conv2(x)))
        # torch.Size([32, 24, 3, 3])
        x = self.pool(x)
        # torch.Size([32, 24, 1, 1])

        x = x.view(x.size(0), -1)
        # torch.Size([32, 24])
        x = F.relu(self.fc1(x))
        # torch.Size([32, 6])

        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()
class DemandPredictorgpt(nn.Module):
    def __init__(self, C, window_size=9):
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
    def __init__(self, C, window_size):
        super(FDemandPredictor_simple, self).__init__()

        self.conv1 = nn.Conv2d(C, 12, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)


        self.fc1 = nn.Linear(32 * (window_size // 2) * (window_size // 2), 6)
        self.fc1 = nn.Linear(32, 6)# Assuming one pooling operation
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



class ScooterDemandPredictor(nn.Module):
    def __init__(self, C):
        super(ScooterDemandPredictor, self).__init__()

        # 1st Convolutional Layer
        self.conv1 = nn.Conv2d(C, 12, kernel_size=3)#, padding=1)
        self.bn1 = nn.BatchNorm2d(12)

        # 2nd Convolutional Layer
        self.conv2 = nn.Conv2d(12, 32, kernel_size=3)#, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 3rd Convolutional Layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)#, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.squeeze()


window_size = 9

time = 'Other'
# cnndf_test.plot(column = 'counts_fstart_' + time, vmax = cnndf_test['counts_fstart_' + time].quantile(0.99))

padded_channels, tensor_targets, width, height, C, ASA_grid = prepare(cnndf_train, traincity, 'counts_fstart_' + time, channels, window_size, 100)
train_loader, val_loader, prediction_data_train = get_modelling_data(ASA_grid, height, width, padded_channels, tensor_targets, window_size, rotations = [0])

# model = JDemandPredictor(C)
# shapes = False#
model = ScooterDemandPredictor(C)
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)
epochs = 8
loops = 4

def run_transfer(model, ASA_grid, loops = 5, lr=0.005, epochs = 10, criterion = criterion ):
    def training_loop(epochs, model, train_loader, val_loader, criterion = criterion, optimizer = optim.Adam(model.parameters(), lr=0.005)):
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
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            # print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {round(train_loss/1):.4f}, Validation Loss: {round(val_loss/1):.4f}")

        return model
    model = training_loop(loops, model, train_loader, val_loader)

    def plot_preds(cnndf, prediction_df, edgecolor = 'red', context = False, alpha =1, city = 'AADO', quantile = 0.95):

        vmax_ = prediction_df[prediction_df['ASA'] == 1].pred.quantile(quantile)#     median() * 2.5
        fig, ax = plt.subplots(figsize=(15, 9))
        remove_labels(ax)
        prediction_df.plot(alpha = alpha, column = 'pred', ax = ax, legend = True, vmax = vmax_) #cmap = 'binary'
        # gpd.read_file('PublicGeoJsons/AmsLines.json').to_crs(28992).plot(ax = ax, linewidth = 0.25)
        cnndf[cnndf['sum_Passengers']>0].plot(ax = ax, facecolor = 'None', edgecolor = 'Yellow')
        cnndf[cnndf['metro']>0].plot(ax = ax, facecolor = 'None', edgecolor = edgecolor)
        if context == True:
            cx.add_basemap(ax, crs=cnndf.crs.to_string(), source=cx.providers.Stamen.Watercolor)
        else:
            if city == 'AADO':
                getAADO(water = True).to_crs(28992).plot(ax=ax, facecolor='None', linewidth=0.05)
            else:
                try:
                    pd.read_pickle('PublicGeoJsons/OtherCities/'+ city + 'PC4.geojson').to_crs(28992).plot(ax=ax, facecolor='None', linewidth=0.15)
                except:
                    gpd.read_file('PublicGeoJsons/OtherCities/'+ city + 'PC4.geojson').to_crs(28992).plot(ax=ax, facecolor='None', linewidth=0.15)

        if 5 == 10:
            cnndf[cnndf['area_intersected_water'] > 0.5].plot(ax=ax, facecolor='None', edgecolor='Blue', linewidth=0.15)
            cnndf[cnndf['area_intersected_park'] > 0.8].plot(ax=ax, facecolor='None', edgecolor='Green', linewidth=0.15)
            cnndf[cnndf['tram'] > 0.8].plot(ax=ax, facecolor='None', edgecolor='Black', linewidth=0.15)
        return
    prediction_df_train = make_predictions(cnndf_train, prediction_data_train, model, ASA_grid)
    # plot_preds(cnndf_train, prediction_df_train, city = traincity)
    # compare_transfer(prediction_df_train)

    padded_channels, tensor_targets, width, height, C, ASA_grid = prepare(cnndf_test, testcity, 'counts_fstart_' + 'Morning',channels, window_size, 100)
    train_loader_test, val_loader_test, prediction_data_test, = get_modelling_data(ASA_grid, height, width, padded_channels, tensor_targets, window_size)
    prediction_df_test = make_predictions(cnndf_test, prediction_data_test, model, ASA_grid)
    # plot_preds(cnndf_test, prediction_df_test, city = testcity)
    # compare_transfer(prediction_df_test)

    def finetune(model, lr=0.005, epochs = epochs, criterion = criterion):

        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last fully connected layer
        for param in model.fc1.parameters():
            param.requires_grad = True

        for param in model.fc2.parameters():
            param.requires_grad = True

        # for param in model.conv3.parameters():
        #     param.requires_grad = True
        #
        # for param in model.bn3.parameters():
        #     param.requires_grad = True
        #
        # for param in model.global_pool.parameters():
        #     param.requires_grad = True

        # Initialize new loss and optimizer for fine-tuning
        criterion = criterion
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


        finetuned_model = training_loop(epochs, model, train_loader_test, val_loader_test)
        return finetuned_model
    finetuned_model = finetune(model)

    prediction_df_test = make_predictions(cnndf_test, prediction_data_test, finetuned_model, ASA_grid)
    plot_preds(cnndf_test, prediction_df_test, city = testcity, quantile = 0.99)
    compare_transfer(prediction_df_test)

    return prediction_df_test
prediction_df_test = run_transfer(model, ASA_grid, epochs = epochs, criterion = criterion, loops = loops)


prediction_df_test.to_pickle('CNN_Demand/CNN_Predictions/Transfer' + str(square_size) + time)
compare(prediction_df_test, val_loader , transit = False, vmax = prediction_df[prediction_df['ASA'] == 1].target.median() * 2.5)
compare(prediction_df, val_loader, val_only= True)

#
# for i in ['Rotterdam', 'AADO', 'Den Haag']:
#     bounds = pd.read_pickle('Demand Modelling/Grids/'+ i+ '/150').total_bounds #.to_crs(epsg=3857)
#     basemap, _ = cx.bounds2img(*bounds, source=cx.providers.CartoDB.Voyager)
#
#     # Save the basemap as an image
#     plt.imsave(i + '.png', basemap, format='png')
#     saved_basemap = mpimg.imread(i + '.png')
#
#     # Overlay the saved basemap on your new plot
#     plt.imshow(saved_basemap)





