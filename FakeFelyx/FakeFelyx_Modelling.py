import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# load data into a pandas DataFrame
data = pd.read_pickle('Embeddings/FakeFelyx/FakeFelyxData')
# data[col] = data[col].astype()
#data = pd.read_pickle('Embeddings/FakeFelyx/FakeFelyxDataEmbedded')
# data = pd.read_pickle('FakeFelyxDataEmbedded')

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





odin = pd.read_pickle('Odin/Odin2Ams')
odin['weekdag'] = odin['weekdag'].apply(lambda x: (x-2)%7)
odin.rename(columns={'weekdag': 'weekday',
                   }, inplace=True, errors='raise')
# odin = odin[['aankpc', 'hour', 'weekday', 'vertpc']]
for column in columns_to_encode:
    odin[column] = odin[column].astype(str)
    odin[column] = label_encoder.fit_transform(odin[column])

odinpred = clf.predict(odin[['aankpc', 'hour', 'weekday', 'vertpc']])
odin['odinpred'] = odinpred
view = odin[['odinpred', 'khvm']]
h = view.groupby(['khvm']).mean()








# with dl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Define the classifier model
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
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
df = data  # Replace 'your_dataset.csv' with the actual file path
new_data = odin[['aankpc', 'hour', 'weekday', 'vertpc']]
for col in new_data.columns:
    new_data[col] = new_data[col].astype(str)
new_data = new_data[new_data['aankpc'].isin(data.aankpc.unique())]
new_data = new_data[new_data['vertpc'].isin(data.vertpc.unique())]

label_encoder = LabelEncoder()
for col in df.columns:
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])
    if col in new_data.columns:
        new_data[col] = label_encoder.transform(new_data[col])

# Split the dataset into train and test
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
model = Classifier(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the classifier
num_epochs = 50
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

# Use the trained model for predictions
model.eval()


new_data['Real'] = list(df['Real'][:len(new_data)])

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
new_data = new_data.join(odin['khvm'])

new_data.groupby(['khvm']).mean()









