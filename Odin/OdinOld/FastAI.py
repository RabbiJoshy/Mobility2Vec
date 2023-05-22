import pandas as pd
import numpy as np
import torch
import torch.nn as nn

df = pd.read_pickle('Odin/Odin2Clean')
dep_var = 'khvm'

categorical_columns = [col for col in df.columns if df.dtypes[col] == object]
categorical_columns.remove(dep_var)
numerical_columns = [col for col in df.columns if df.dtypes[col] != object]

# Define the input features
cat_cols_1 = categorical_columns[:8]
cat_cols_2 = categorical_columns[8:]
# cat_cols_1 = ['cat_feature_1', 'cat_feature_2']  # Categorical features to be embedded in the first subset
# cat_cols_2 = ['cat_feature_3', 'cat_feature_4',
#               'cat_feature_5']  # Categorical features to be embedded in the second subset
# num_cols = ['num_feature_1', 'num_feature_2', 'num_feature_3', 'num_feature_4', 'num_feature_5']  # Numerical features
num_cols = numerical_columns
num_features = len(num_cols)
num_categorical_features_1 = len(cat_cols_1)
num_categorical_features_2 = len(cat_cols_2)
embedding_dim = 4  # Embedding dimension

# Load the dataframe

# Define the embedding layers
embedding_layer_1 = nn.Embedding(num_embeddings=df[cat_cols_1[0]].nunique(), embedding_dim=embedding_dim)
embedding_layer_2 = nn.Embedding(num_embeddings=df[cat_cols_2[0]].nunique(), embedding_dim=embedding_dim)


# Define the deep neural network model
class DNN(nn.Module):
    def __init__(self, cat_cols_1, cat_cols_2, num_cols, emb_dim=10, hidden_dim=50, num_classes=5):
        super(DNN, self).__init__()
        self.emb_layers = nn.ModuleList()
        self.cat_cols_1 = cat_cols_1
        self.cat_cols_2 = cat_cols_2
        self.num_cols = num_cols
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Embedding layers for categorical columns
        for col in cat_cols_1:
            num_embeddings = df[col].nunique()
            emb_layer = nn.Embedding(num_embeddings, emb_dim)
            self.emb_layers.append(emb_layer)

        for col in cat_cols_2:
            num_embeddings = df[col].nunique()
            emb_layer = nn.Embedding(num_embeddings, emb_dim)
            self.emb_layers.append(emb_layer)

        # Linear layers for numerical columns
        num_inputs = len(cat_cols_1) * emb_dim + len(cat_cols_2) * emb_dim + len(num_cols)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_cat1, x_cat2, x_num):
        # Categorical columns
        x_emb_list = []

        for i, col in enumerate(self.cat_cols_1):
            x_emb_list.append(self.emb_layers[i](x_cat1[:, i]))

        for i, col in enumerate(self.cat_cols_2):
            x_emb_list.append(self.emb_layers[i + len(self.cat_cols_1)](x_cat2[:, i]))

        x_cat = torch.cat(x_emb_list, 1)

        # Numerical columns
        x_num = x_num.float()

        # Concatenate categorical and numerical inputs
        x = torch.cat((x_cat, x_num), dim=1)

        # Pass through linear layers
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.Softmax(dim=1)(x)
        return x

# Instantiate the model
model = DNN(num_features, num_categorical_features_1, num_categorical_features_2, embedding_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 2
batch_size = 32
num_batches = int(np.ceil(len(df) / batch_size))

for epoch in range(num_epochs):
    for batch in range(num_batches):
        # Extract the batch from the dataframe
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        x_categorical_1 = torch.tensor(batch_df[cat_cols_1].astype(int).values)
        x_categorical_2 = torch.tensor(batch_df[cat_cols_2].astype(int).values)
        x_numerical = torch.tensor(batch_df[num_cols].astype(np.float32).values)
        y = torch.tensor(batch_df[dep_var].astype(int).values)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(x_categorical_1, x_categorical_2, x_numerical)
        # Compute the loss
        loss = criterion(outputs, y)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Print the loss after each epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

cat_1 = torch.tensor([[2, 1, 3, 4, 2, 1, 3, 4], [1, 2, 4, 3, 1, 2, 3, 4]])  # Two samples, 8 categorical features
cat_2 = torch.tensor([[3, 0, 2, 1, 0, 1, 3, 2, 0, 3, 2, 1, 0, 3, 1, 2], [0, 2, 1, 3, 1, 3, 0, 2, 3, 2, 0, 1, 3, 2, 1, 0]])  # Two samples, 16 categorical features
num = torch.tensor([[1.2, 3.4, 5.6, 7.8, 9.0, 1.2, 3.4, 5.6, 7.8, 9.0, 1.2, 3.4, 5.6, 7.8, 9.0, 1.2, 3.4], [5.6, 7.8, 1.2, 3.4, 9.0, 5.6, 7.8, 1.2, 3.4, 9.0, 5.6, 7.8, 1.2, 3.4, 9.0, 5.6, 7.8]])  # Two samples, 18 numerical features

# Pass the input through the model
output = model(cat_1, cat_2, num)