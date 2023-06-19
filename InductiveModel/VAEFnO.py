import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colormaps
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import geopandas as gpd
import matplotlib.pyplot as plt
import json
import os
from dfencoder import AutoEncoder, load_model
VAE_dimensionality = 8
Path = 'InductiveModel'
Path = os.path.join(Path, 'Ams')
def readydf(KHVM = True):

 with open('Features_Dictionary', "r") as json_file:
  fd = json.load(json_file)

 df = pd.read_pickle(os.path.join(Path, 'Raw'))


 cols = []
 cols = fd['FelyxKnown']
 cols += fd['OTP']
 cols += fd['Location']
 cols += fd['Time']
 # # cols += fd['Comparison']
 cols += fd['PCInfo']
 cols += fd['Weather']
 if KHVM == True:
     cols += fd['Targets']

 cols = [x for x in cols if x not in ['khvm']]#, 'oprijbewijsau', 'feestdag']]
 df = df[cols]
 print(df.head())

 return df
def remodel(df):
 model = load_model(os.path.join(Path, str(VAE_dimensionality)))
 FullEmbeddingMatrix = model.get_deep_stack_features(df)
 print(FullEmbeddingMatrix.shape)
 EmbeddingMatrix = FullEmbeddingMatrix[:, -(VAE_dimensionality):].numpy()
 df[['emb' + str(x) for x in range(VAE_dimensionality)]] = EmbeddingMatrix
 return df
df = readydf()
# df = df.replace({'hvm': {'Felyx': 'Personenauto'}})
# df = df.replace({'jaar': {'2023': '2018'}})
# df = df.replace({'aankpc': {'Felyx': '2050'}})
# df = df.replace({'vertpc': {'Felyx': '2050'}})
# df = df.replace({'hvm': {'Felyx': 'Niet-elektrische fiets'}})
# df = df.replace({'weekdag': {'Woensdag': 'Dinsdag'}})
df = remodel(df)
df['khvm'] = pd.read_pickle(os.path.join(Path, 'Raw'))['khvm']
df['aankpc'] = pd.read_pickle(os.path.join(Path, 'Raw'))['aankpc']
PCs = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
APC = [str(x) for x in PCs['Postcode4']]
df = df[df.aankpc.isin(APC)]
n_components=3

def ReduceDF(df, reductionmethod = 'PCA', fit = None):
 if fit:
  pca = fit
 else:
  if reductionmethod == 'PCA':
   pca = PCA(n_components)
  else:
   df = df.sample(5000)
   pca = TSNE(n_components, perplexity = 50)
 df[['PC' + str(x) for x in range(n_components)]] = pca.fit_transform(df[['emb' + str(x) for x in range(VAE_dimensionality)]])
# sns.scatterplot(data = df.sample(300), x = 'PC0', y = 'PC1', hue = 'khvm')
 return df, pca
df, pca = ReduceDF(df)
def plotodin(Odin, colorcol='choice', centers=True, filter = False):
 if filter != False:
  Odin = Odin[Odin[colorcol].isin(filter)]

 colormap = colormaps.get_cmap('tab20')  # , len(show[colorcol].unique()))
 color_dict = {category: colormap(i) for i, category in enumerate(Odin[colorcol].unique())}

 fig = plt.figure(figsize=(8, 6))
 ax = fig.add_subplot(111, projection='3d')
 if type(Odin[colorcol].iloc[0]) == str:
  p = ax.scatter(*[Odin['PC' +str(x)] for x in range(3)], c=Odin[colorcol].map(color_dict))

  for category, color in color_dict.items():
   ax.plot([], [], 'o', color=color, label=category[:10])
   ax.legend()

 else:
  p = ax.scatter(*[Odin['PC' +str(x)] for x in range(3)], c=Odin[colorcol])
  plt.colorbar(p)

 ax.set_xlabel('PC0')
 ax.set_ylabel('PC1')
 ax.set_zlabel('PC2')
 ax.set_title(colorcol)

 plt.show()
plotodin(df.sample(500), colorcol = 'khvm')
plotodin(df.sample(2500), colorcol = 'khvm', filter = ['Te voet', 'Trein', 'Personenauto - bestuurder', 'Bus/tram/metro'])
# plotodin(df.sample(1500), colorcol = 'hvm', filter = ['Elektrische fiets', 'Niet-elektrische fiets','Personenauto'])
# plotodin(df.sample(1000), colorcol = 'hvm', filter = ['Elektrische fiets', 'Felyx','Personenauto'])
# pcadf = pd.DataFrame(pca.components_, columns = pca.feature_names_in_)
#
# see = df[df.khvm !='Felyx']
# felonly = df[df.khvm == 'Felyx']
df.khvm.unique()
#
corr = pd.get_dummies(df.drop(['weekdag', 'aankpc', 'vertpc'], axis = 1)).corr()[['PC0', 'PC1', 'PC2']]






def xclassify(df, s = 0):
 if s == 0:
  s = len(df)
 df = df.sample(s)
 model = xgb.XGBClassifier()
 target = 'khvm'
 encoder = LabelEncoder()
 encoder.fit(df[target])
 df[target] = encoder.transform(df[target])

 X = df[['emb' + str(x) for x in range(VAE_dimensionality)]]
 y = df[target]
 X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3)

 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 print(accuracy_score(y_test, y_pred))

 X_test['predictions'] = encoder.inverse_transform(y_pred)
 X_test[target] = encoder.inverse_transform(y_test)
 types = encoder.inverse_transform(list(range(len(y_test.unique()))))
 print(classification_report(X_test.predictions, X_test[target], target_names=types))
 X_test[target] = encoder.inverse_transform(y_test)

 return
xclassify(df)