import pandas as pd
import torch
from torch import nn
from torch.autograd.variable import Variable
from sklearn.preprocessing import StandardScaler


# Define generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_noise, 256),
            nn.ReLU(),
            nn.Linear(256, n_features),
        )

    def forward(self, input):
        return self.main(input)


# Define discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()

    # Train on Real Data
    prediction_real = discriminator(real_data)
    error_real = criterion(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # Train on Fake Data
    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()

    prediction = discriminator(fake_data)
    error = criterion(prediction, real_data_target(fake_data.size(0)))
    error.backward()

    optimizer.step()

    return error


# Load your dataframe
df = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAmsterdam').sample(1000)
df.columns
df = df[[ 'pt_duration', 'pt_distance','walk_duration', 'walk_distance',
       'bike_duration', 'bike_distance', 'car_duration', 'car_distance','sin_time', 'cos_time']]

# Preprocess your data
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Convert DataFrame into PyTorch tensor
data = torch.tensor(df.values, dtype=torch.float32)

# Initialize generator and discriminator
n_features = 10
n_noise = 100
generator = Generator()
discriminator = Discriminator()

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)

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
fake_df = pd.DataFrame(fake_data.numpy(), columns=df.columns)

# If you've scaled your data, you might want to reverse the scaling operation
fake_df = pd.DataFrame(scaler.inverse_transform(fake_df), columns=fake_df.columns)

# Save the generated data to a CSV file
fake_df.to_csv('fake_data.csv', index=False)
