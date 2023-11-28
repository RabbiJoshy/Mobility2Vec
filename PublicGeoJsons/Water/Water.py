import requests
import json
from ModellingUtilities import getAADO
from PlotUtilities import *
import requests
import time

city = 'AADO'
square_size = 150
grid = pd.read_pickle('Demand Modelling/Grids/' + city + '/' + str(square_size))
# Get the bounding box
bbox = grid.total_bounds  # returns a tuple (minx, miny, maxx, maxy)
# Convert bounding box to POLYGON WKT format
bbox_wkt = f"POLYGON(({bbox[0]} {bbox[1]}, {bbox[0]} {bbox[3]}, {bbox[2]} {bbox[3]}, {bbox[2]} {bbox[1]}, {bbox[0]} {bbox[1]}))"


# Initial POST request
post_url = "https://api.pdok.nl/brt/top10nl/download/v1_0/full/custom"
post_data = {
    "featuretypes": ["waterdeel"],
    "format": "gml",
    "geofilter": bbox_wkt
}
headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

response = requests.post(post_url, json=post_data, headers=headers)
status_url = response.json()["_links"]["status"]["href"]

# Complete URL for status
complete_status_url = f"https://api.pdok.nl{status_url}"

# Wait and check for completion
while True:
    status_response = requests.get(complete_status_url, headers={'accept': 'application/json'})
    if status_response.json()["status"] == "COMPLETED":
        download_url = f"https://api.pdok.nl{status_response.json()['_links']['download']['href']}"
        break
    time.sleep(2)

# Download the ZIP file
response = requests.get(download_url)
with open('PublicGeoJsons/water/' + city + '.zip', 'wb') as f:
    f.write(response.content)

db = gpd.read_file('PublicGeoJsons/water/' + 'top10nl_waterdeel.gml' , driver='GML')
db.set_crs(28992).to_pickle('PublicGeoJsons/water/'+ 'AADO')

db = gpd.read_file('PublicGeoJsons/water/'+ 'Den Haag')
db = pd.read_pickle('PublicGeoJsons/water/'+ 'AADO')
db = db.loc[db['geometry'].geom_type.isin(['Polygon', 'MultiPolygon'])].set_crs(28992)
db.plot()


for i in db.visualisatieCode.unique():
    fig, ax = plt.subplots()
    # getAADO().to_crs(28992).plot(ax = ax, facecolor = 'None', linewidth = 0.1)
    db[db.visualisatieCode == i].plot(ax = ax, alpha = 0.5)
    remove_labels(ax)
    ax.set_title(i)

# 12500, 12430, 12420 (12200, 12300?)
#
# db[db.visualisatieCode == 12200].plot(linewidth = 0.1)