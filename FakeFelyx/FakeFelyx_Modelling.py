import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data = pd.read_pickle('FakeFelyx/FakeFelyxData')
odin = pd.read_pickle('Odin/Odin2Ams')
odin['weekdag'] = odin['weekdag'].apply(lambda x: (x-2)%7)
odin.rename(columns={'weekdag': 'weekday',
                   }, inplace=True, errors='raise')

odin.columns

def modelxgb():
    columns_to_encode = list(data.columns)
    columns_to_encode.remove('Real')
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        data[column] = data[column].astype(str)
        data[column] = label_encoder.fit_transform(data[column])

    # split data into training and testing sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('Real', axis=1), data['Real'], test_size=0.2, random_state=42)
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    X_test['pred'] = y_pred
    X_test['Real'] = y_test
    odinview = X_test

    for column in columns_to_encode:
        odin[column] = odin[column].astype(str)
        odin[column] = label_encoder.fit_transform(odin[column])

    odinpred = clf.predict(odin[['aankpc', 'hour', 'weekday', 'vertpc']])
    odin['odinpred'] = odinpred
    view = odin[['odinpred', 'khvm', 'aankpc', 'hour', 'weekday', 'vertpc']]
    h = view.groupby(['khvm']).mean()

    return view, h, odinview

view, h, odinview = modelxgb()

# with dl

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx, :-1], dtype=torch.float32)
        y = torch.tensor(self.data[idx, -1], dtype=torch.float32)
        return x, y

# Load the dataset
# Load the dataset
df = data  # Replace 'your_dataset.csv' with the actual file path
new_data = odin[['aankpc', 'hour', 'weekday', 'vertpc']]
for col in new_data.columns:
    new_data[col] = new_data[col].astype(str)
    data[col] = data[col].astype(str)

new_data = new_data[new_data['aankpc'].isin(data.aankpc.unique())]
new_data = new_data[new_data['vertpc'].isin(data.vertpc.unique())]

label_encoder = LabelEncoder()
for col in df.columns:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])
    if col in new_data.columns:
        new_data[col] = label_encoder.transform(new_data[col])

# def initialise:()
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Create dataloaders
train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the classifier
input_size = len(df.columns) - 1
hidden_size = 16
hidden_size2 = 6
model = Classifier(input_size, hidden_size, hidden_size2)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the classifier
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset)

    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    test_loss /= len(test_dataset)
    accuracy = correct / len(test_dataset)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')









model.eval()
new_data['Real'] = [0.5] * len(new_data)#list(df['Real'][:len(new_data)])

# Create a dataloader for the new data
new_dataset = CustomDataset(new_data)
new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False)

# Perform predictions on the new data
predictions = []
with torch.no_grad():
    for inputs, _ in new_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = torch.round(outputs)
        predictions.extend(predicted.squeeze().tolist())

new_data['pred'] = predictions



# odin.khvm = odin.khvm.astype(str)
# odin = odin.replace({"khvm": {'3':'4'}})
# odin = odin.replace({"khvm": {'2':'1'}})
# odin = odin[odin.khvm != '7']
choice_dict = pd.read_json('Odin/OdinData/odin-col-dict.json')
odin = odin.replace({"khvm": choice_dict['khvm']})


kvhm = new_data.join(odin['khvm'])
kvhm.groupby(['khvm']).mean()

new_data.pred.value_counts()


kvhm.to_pickle('false_predictions')









