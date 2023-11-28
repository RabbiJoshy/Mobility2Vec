import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.autograd.variable import Variable
from ModellingUtilities import *
df = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAADO').dropna()
Amenities = pd.read_pickle('SAO/OSM/Amenities/Felyx')
AmenitiesRed = reduce_df(Amenities, 3)
df = df.join(AmenitiesRed)
df = df[['pt_duration', 'pt_distance',  'walk_duration',
       'walk_distance', 'bike_duration', 'bike_distance',
       'car_duration', 'car_distance', 'felyx_cost', 'pt_cost', 'car_cost', 'windspeed',
       'temp', 'feelslike', 'precip', 'precipcover', 'PC0', 'PC1', 'PC2', 'time_category']]

n_continuous = 16
n_categorical = 1
n_noise = 1

# Define the generator and discriminator
class Generator(nn.Module):
    def __init__(self, n_input, n_out_cont, n_out_cat):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2_cont = nn.Linear(256, n_out_cont)
        self.fc2_cat = nn.Linear(256, n_out_cat)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        cont_data = torch.tanh(self.fc2_cont(x)) # Using tanh to bound the output between -1 and 1
        cat_data = torch.softmax(self.fc2_cat(x), dim=-1)
        return torch.cat([cont_data, cat_data], dim=1)

class Discriminator(nn.Module):
    def __init__(self, n_input):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# Preprocess your data
scaler = StandardScaler()
ohe = OneHotEncoder(sparse=False)

# Assume that the first 10 columns are continuous and the last 2 are categorical
df_continuous = df.iloc[:, :16]
df_categorical = df.iloc[:, 16:]

# Convert categorical variables into one-hot encoding
df_categorical_ohe = pd.DataFrame(ohe.fit_transform(df_categorical))
feature_names = [f"x{i}_{c}" for i, categories in enumerate(ohe.categories_) for c in categories]
df_categorical_ohe.columns = feature_names

# Standardize the continuous variables
df_continuous_scaled = pd.DataFrame(scaler.fit_transform(df_continuous), columns=df_continuous.columns)

# Concatenate the preprocessed continuous and categorical data
df_processed = pd.concat([df_continuous_scaled, df_categorical_ohe], axis=1)

# Convert DataFrame into PyTorch tensor
data = torch.tensor(df_processed.values, dtype=torch.float32)

# Define the number of continuous and one-hot encoded categorical features
n_out_cont = len(df_continuous_scaled.columns)
n_out_cat = len(df_categorical_ohe.columns)
n_input = n_out_cont + n_out_cat
n_noise = 1  # Set this to the desired noise dimension

# Create instances of the generator and the discriminator.
generator = Generator(n_noise, n_out_cont, n_out_cat)
discriminator = Discriminator(n_input)

# Define the loss function
loss_function = nn.BCELoss()

# Define the optimizers
learning_rate = 0.001
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Define the training functions for the discriminator and the generator
def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()

    # Predict using real data
    prediction_real = discriminator(real_data)
    error_real = loss_function(prediction_real, Variable(torch.ones(real_data.size(0), 1)))
    error_real.backward()

    # Predict using fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss_function(prediction_fake, Variable(torch.zeros(fake_data.size(0), 1)))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()

    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    error = loss_function(prediction, Variable(torch.ones(fake_data.size(0), 1)))
    error.backward()

    optimizer.step()

    return error


# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    real_data = data

    # Generate fake data
    noise = torch.randn(real_data.size(0), n_noise)
    fake_data = generator(noise)

    # Train discriminator
    d_error = train_discriminator(optimizer_d, real_data, fake_data.detach())

    # Generate fresh fake data
    noise = torch.randn(real_data.size(0), n_noise)
    fake_data = generator(noise)

    # Train generator
    g_error = train_generator(optimizer_g, fake_data)

    # Log errors and print the error of the last epoch
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}:\tDiscriminator Error: {d_error}\t\tGenerator Error: {g_error}")

# Generate a batch of fake data after training
noise = torch.randn(real_data.size(0), n_noise)
fake_data = generator(noise).detach()

# Convert the tensor back into a DataFrame
fake_df = pd.DataFrame(fake_data.numpy(), columns=df_processed.columns)

# Split the data back into continuous and categorical
fake_df_continuous = fake_df.iloc[:, :n_continuous]
fake_df_categorical = fake_df.iloc[:, n_continuous:]

# Inverse transform continuous data
fake_df_continuous = pd.DataFrame(scaler.inverse_transform(fake_df_continuous), columns=fake_df_continuous.columns)

# Inverse transform categorical data (one-hot encoded data)
# This will give you the category index (you may want to map this back to category name using the original label encoder if you used it)
fake_df_categorical = pd.DataFrame(ohe.inverse_transform(fake_df_categorical), columns=df_categorical.columns)

# Concatenate the dataframes to get the final dataframe
final_fake_df = pd.concat([fake_df_continuous, fake_df_categorical], axis=1)

# Save the generated data to a CSV file
final_fake_df.to_csv('fake_data.csv', index=False)
