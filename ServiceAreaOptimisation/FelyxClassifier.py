from ModellingUtilities import *
from sklearn.cluster import KMeans
import numpy as np
f = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAADO')
numeric_cols = f.select_dtypes(include=[np.number]).columns
Amenities = pd.read_pickle('SAO/OSM/Amenities/AADO')
AmenitiesRed = reduce_df(Amenities, 3)

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(AmenitiesRed)
df = AmenitiesRed
df['cluster'] = kmeans.labels_

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection ="3d")

# Create a color map dictionary
cmap = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}  # replace with actual cluster names and desired colors

scatter = ax.scatter3D(df['PC0'], df['PC1'], df['PC2'], c=df['cluster'].map(cmap))

plt.title("3D scatter plot")
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')
plt.show()

