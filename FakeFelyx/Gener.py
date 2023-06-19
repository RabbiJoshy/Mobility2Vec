from shapely.geometry import MultiPoint, Point
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

def genservicearea(x = 200):
    odin2019 = pd.read_pickle('FelyxData/Raw Movement/JA23Movers')
    multipoint = MultiPoint(odin2019.geometry.iloc[:x].values)
    # Get the convex hull of the MultiPoint, which will be a Polygon
    polygon = multipoint.convex_hull
    fig, ax = plt.subplots()
    buu = gpd.read_file('PublicGeoJsons/Amsbuurts.json')
    buu.plot(ax = ax, facecolor = 'None')
    g = gpd.GeoSeries(polygon)
    g.plot(ax = ax, facecolor = 'None', edgecolor = 'blue')
    return polygon
test_polygon = genservicearea()

dep_var = 'khvm'
cols = ['vertpc', 'aankpc', 'khvm', 'weekdag',
       'bike_dur', 'bike_dist', 'car_dur',
       'car_dist', 'pt_dur', 'pt_dist', 'walk_dur', 'walk_dist', 'sin_time',
        'cos_time','choice_dur', 'choice_dist','oprijbewijsau','feestdag'] #'hour'
cols += ['prev_time']
# cols += ['aank Totaal', 'aank Man', 'aank Vrouw', 'aank tot 15 jaar',
#    'aank 15 tot 25 jaar', 'aank 25 tot 45 jaar', 'aank 45 tot 65 jaar',
#    'aank 65 jaar en ouder',
#    'aank Geboren in Nederland met een Nederlandse herkomst',
#    'aank Geboren in Nederland met een Europese herkomst (excl. Nederland)',
#    'aank Geboren in Nederland met herkomst buiten Europa',
#    'aank Geboren buiten Nederland met een Europese herkomst (excl. Nederland)',
#    'aank Geboren buiten Nederland met een herkomst buiten Europa',
#    'aank Totaal.1', 'aank Eenpersoons',
#    'aank Meerpersoons \nzonder kinderen', 'aank Eenouder',
#    'aank Tweeouder', 'aank Huishoudgrootte', 'aank Totaal.2',
#    'aank voor 1945', 'aank 1945 tot 1965', 'aank 1965 tot 1975',
#    'aank 1975 tot 1985', 'aank 1985 tot 1995', 'aank 1995 tot 2005',
#    'aank 2005 tot 2015', 'aank 2015 en later', 'aank Meergezins',
#    'aank Koopwoning', 'aank Huurwoning', 'aank Huurcoporatie',
#    'aank Niet bewoond', 'aank WOZ-waarde\nwoning',
#    'aank Personen met WW, Bijstand en/of AO uitkering\nBeneden AOW-leeftijd',
#    'aank density', 'vert Totaal', 'vert Man', 'vert Vrouw',
#    'vert tot 15 jaar', 'vert 15 tot 25 jaar', 'vert 25 tot 45 jaar',
#    'vert 45 tot 65 jaar', 'vert 65 jaar en ouder',
#    'vert Geboren in Nederland met een Nederlandse herkomst',
#    'vert Geboren in Nederland met een Europese herkomst (excl. Nederland)',
#    'vert Geboren in Nederland met herkomst buiten Europa',
#    'vert Geboren buiten Nederland met een Europese herkomst (excl. Nederland)',
#    'vert Geboren buiten Nederland met een herkomst buiten Europa',
#    'vert Totaal.1', 'vert Eenpersoons',
#    'vert Meerpersoons \nzonder kinderen', 'vert Eenouder',
#    'vert Tweeouder', 'vert Huishoudgrootte', 'vert Totaal.2',
#    'vert voor 1945', 'vert 1945 tot 1965', 'vert 1965 tot 1975',
#    'vert 1975 tot 1985', 'vert 1985 tot 1995', 'vert 1995 tot 2005',
#    'vert 2005 tot 2015', 'vert 2015 en later', 'vert Meergezins',
#    'vert Koopwoning', 'vert Huurwoning', 'vert Huurcoporatie',
#    'vert Niet bewoond', 'vert WOZ-waarde\nwoning',
#    'vert Personen met WW, Bijstand en/of AO uitkering\nBeneden AOW-leeftijd',
#    'vert density']
rawdf = pd.read_pickle('Odin/OdinModellingData/Odin2019All')[cols]
felyx = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAmsterdam')
APCdf = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
APC = [str(PC) for PC in list(gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')['Postcode4'].unique())]
felyx['khvm'] = ['Felyx']* len(felyx)
felyx = felyx[cols]
felyx['Felyx'] = [1]* len(felyx)
rawdf['Felyx'] = [0]* len(rawdf)
amsdf = rawdf[rawdf.aankpc.isin(APC) & rawdf.vertpc.isin(APC)]

intersects = APCdf[~APCdf['geometry'].intersects(test_polygon)]
intersects.Postcode4 = intersects.Postcode4.astype(str)
buitenstad = rawdf[rawdf.aankpc.isin(intersects.Postcode4.unique()) |rawdf.vertpc.isin(intersects.Postcode4.unique())]


# data = pd.concat([felyx.sample(10000), amsdf, rawdf.sample(3000)]).drop('khvm', axis = 1)
data = pd.concat([felyx, rawdf.sample(25000)])
def ready(data):
    data = data.drop('khvm', axis = 1)
    data = data.drop(['aankpc', 'vertpc', 'choice_dur', 'choice_dist'], axis = 1)
    inf = pd.to_datetime(data['prev_time'],format= '%H:%M' )
    data['seconds'] = inf.dt.second + 60*(inf.dt.minute) +300*(inf.dt.hour)
    def round_to_multiple(number, multiple):
        return round(number / multiple) * multiple
    seconds_in_day = 24*60*60
    # Apply the function to the 'Numbers' column
    data['seconds'] = data['seconds'].apply(lambda x: round_to_multiple(x, 900))
    data['sin_time'] = np.sin(2*np.pi*data.seconds/seconds_in_day)
    data['cos_time'] = np.cos(2*np.pi*data.seconds/seconds_in_day)
    data = data.drop(['seconds', 'prev_time'], axis = 1)
    return data
data = ready(data)
buitendata = ready(buitenstad)

# data = data[['sin_time','cos_time', 'Felyx']]

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('Felyx', axis=1), data['Felyx'], test_size=0.33, random_state=42)

cat_cols = X_train.select_dtypes(include=['object']).columns

from sklearn import preprocessing
for col in cat_cols:
    print(col)
    le = preprocessing.LabelEncoder()
    le.fit(list(X_train[col]) + list(X_test[col]))
    X_test[col] = le.transform(X_test[col])
    X_train[col] = le.transform(X_train[col])
    buitendata[col] = le.transform(buitendata[col])
y_train = le.fit_transform(y_train)


model = xgb.XGBClassifier()
model.fit(X_train, y_train) #sample_weight=sample_weights
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# xgb.plot_importance(model, importance_type = 'weight') #

X_test['predictions'] = y_pred
X_test['Felyx'] = le.inverse_transform(y_test)

corr = X_test.drop('weekdag', axis =1).corr()

X_test['error'] = X_test['Felyx'] != X_test['predictions']

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

buitendata['prediction']= model.predict(buitendata.drop(['Felyx'], axis = 1))
new = buitendata.join(rawdf[['aankpc', 'vertpc']])
newfel = new[new.prediction==1]



