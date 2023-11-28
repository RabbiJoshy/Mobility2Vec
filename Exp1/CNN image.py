import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
grid_maps = [np.random.rand(10, 10) for _ in range(8)]
width , height = 93,98
grid_maps = [cnndf_train[channel].values.reshape(width, height) for channel in channels]

import matplotlib.colors as mcolors

# Create custom colormap from white to blue
colors = [(1, 1, 1), (0, 0, 1)]  # from white to blue
n_bins = 100
cmap_name = 'white_to_blue'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.axis('off')

for i in range(len(grid_maps)):
    x, y = np.meshgrid(np.linspace(0, grid_maps[i].shape[1] - 1, grid_maps[i].shape[1]),
                       np.linspace(0, grid_maps[i].shape[0] - 1, grid_maps[i].shape[0]))

    z = np.full_like(x, i * 15)

    norm = plt.Normalize(grid_maps[i].min(), grid_maps[i].max())

    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=custom_cmap(norm(grid_maps[i])), shade=False)

plt.show()
