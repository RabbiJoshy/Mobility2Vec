from ModellingUtilities import *
from PlotUtilities import *
import json

with open('PublicGeoJsons/Parks/Rotterdam Open Space.json') as json_file:
    data = json.load(json_file)
l = data['layers']
