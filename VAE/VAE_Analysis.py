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
with open('Odin/Explanations/otpexpl.json', "r") as json_file:
 choice_dict = json.load(json_file)
encoder = LabelEncoder()
APC = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
VAE_dimensionality = 10
n_components=3
df = pd.read_pickle('VAE/OldNoVerplid/WithoutKHVM/Odin/E10')
odin2019 = pd.read_pickle('Odin/OdinModellingData/Odin2019All')
odin2019 = odin2019.sort_index()
extra = pd.read_pickle('Odin/OldWithVerplid/odin2019sql')
df.index = odin2019.index
hvmexpl = choice_dict['hvm']
df['khvm'] = odin2019['khvm']
extra.hvm = extra.hvm.astype(int).astype(str)
extra = extra.replace({"hvm": choice_dict['hvm']})
extra.hvm = extra.hvm.astype(str)
df = df.join(extra[['hvm', 'windspeed', 'temp', 'feelslike',
       'description']], how = 'inner')

#TODO - fix odin columns to be categorical, get 2020 data to match

grouped = df.groupby(['khvm','hvm']).count()
# df = df[df.hvm.isin(list(grouped[grouped.vertpc > 50].reset_index().hvm))]

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
   ax.plot([], [], 'o', color=color, label=category[:7])
   ax.legend()

 else:
  p = ax.scatter(*[Odin['PC' +str(x)] for x in range(3)], c=Odin[colorcol])
  plt.colorbar(p)

 ax.set_xlabel('PC0')
 ax.set_ylabel('PC1')
 ax.set_zlabel('PC2')
 ax.set_title(colorcol)

 plt.show()
plotodin(df.sample(1000), colorcol = 'khvm')
# plotodin(df, colorcol = 'hvm', filter = ['Snorfiets', 'Bromfiets', 'Bestuurder'])
# colorcolview = 'leeftijd'
# plotodin(df.join(odin2019[colorcolview]).sample(300), colorcol = colorcolview)
def xgc(df, samples =0, reduce = 0, target = 'hvm', weather = False):

 ledict = dict()
 for i in ['windspeed', 'temp', 'feelslike', 'description', target]:
  encoder = LabelEncoder()
  encoder.fit(df[i])
  ledict[i] = encoder
  df[i] = encoder.transform(df[i])

 if samples > 0:
  df = df.sample(samples, random_state=42)

 model = xgb.XGBClassifier()
 if weather == True:
  X = df[['emb' + str(x) for x in range(10)] + ['windspeed', 'temp', 'feelslike','description']]
 else:
  X = df[['emb' + str(x) for x in range(10)]]
 if reduce > 0:
  X = df[['PC' + str(x) for x in range(3)]]
 y = df[target]

 X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=28)

 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)

 for i in ['windspeed', 'temp', 'feelslike', 'description']:
  df[i] = ledict[i].inverse_transform(df[i])
  X_test[i] = encoder.inverse_transform(y_test)

 X_test['predictions'] = ledict[target].inverse_transform(y_pred)
 X_test[target] = ledict[target].inverse_transform(y_test)

 print(accuracy_score(y_test, y_pred))
 types = encoder.inverse_transform(list(range(len(y_test.unique()))))
 print(classification_report(X_test.predictions, X_test[target], target_names=types))
 cm = confusion_matrix(X_test.predictions, X_test[target])
 cmdf = pd.DataFrame(cm, index=types, columns=types)

 return X_test, model, cmdf

TestDF, classifier, confus = xgc(df, target = 'hvm')
TestDF2, classifier, confus2 = xgc(df, target = 'khvm', weather = True)


cm2 = confusion_matrix(TestDF2.predictions, TestDF2.khvm)
cmdf2 = pd.DataFrame(cm2, index = types2, columns = types2)
print(classification_report(TestDF2.predictions, TestDF2.khvm, target_names= types2))

def amsify(tdf):
 tdf = tdf.join(df[['aankpc', 'vertpc']])
 amsdf = tdf[tdf.aankpc.isin([str(x) for x in APC.Postcode4.unique()]) & tdf.vertpc.isin([str(x) for x in APC.Postcode4.unique()])]
 return amsdf

aall = amsify(df.drop(['aankpc', 'vertpc'], axis = 1))
aall.khvm.value_counts()
a2 = amsify(TestDF2)
a1 = amsify(TestDF)
def makecm(df, typ):
 cm = confusion_matrix(df.predictions, df.khvm)
 cmdf = pd.DataFrame(cm, index=typ, columns=typ)
 return cmdf

cc2 = makecm(a2, types2)
cc = makecm(a1, types)





amsdf, _ = ReduceDF(amsdf, fit = pca)
# _, amsclf, amsenc, amscm = xgc(amsdf)
plotodin(amsdf, colorcol = 'khvm')

#Felyx
felyx = pd.read_pickle('VAE/OldNoVerplid/WithoutKHVM/Felyx/E10')
preds = classifier.predict(felyx[['emb' + str(x) for x in range(10)]])
felyx['khvm'] = encoder.inverse_transform(preds)
felyx['khvm'].value_counts()
felyx, _ = ReduceDF(felyx, fit = pca)

def plotboth(felyx, Odin, colorcol = 'khvm'):
 colormap = colormaps.get_cmap('tab20')  # , len(show[colorcol].unique()))
 types = list(set(felyx['khvm'].unique()).union(set(Odin['khvm'].unique())))
 print(types)
 color_dict = {category: colormap(i) for i, category in enumerate(types)}


 fig = plt.figure(figsize=(8, 6))
 ax = fig.add_subplot(111, projection='3d')
 ax.scatter(*[Odin['PC' + str(x)] for x in range(3)], c=Odin['khvm'].map(color_dict), marker="$0$")
 ax.scatter(*[felyx['PC' + str(x)] for x in range(3)], c=felyx['khvm'].map(color_dict), marker="$f$")

 for category, color in color_dict.items():
  ax.plot([], [], 'o', color=color, label=category)
  ax.legend()

 ax.set_xlabel('PC0')
 ax.set_ylabel('PC1')
 ax.set_zlabel('PC2')
 ax.set_title(colorcol)
 return
plotboth(felyx.sample(1000), amsdf)
plotboth(felyx[felyx.khvm ==  'Te voet'] ,amsdf[amsdf.khvm ==  'Te voet'] )


odin2019.doel = odin2019.doel.astype(str)
_, classifier, encoder, cm = xgc(df.join(odin2019.doel), 5000, target = 'doel')







Treinonly = Fel[Fel.prediction == 'Trein'][['aankpc', 'vertpc', 'weekdag', 'sin_time']]
Treinonly.aankpc.unique()

otpfelyx = pd.read_pickle('FelyxData/FelyxModellingData/FelyxOTP')

Treinonly = Treinonly.join(otpfelyx[['lat', 'lon', 'prev_location']])

fig, ax = plt.subplots()
APC.plot(ax = ax, facecolor = 'None')
for i in range(len(Treinonly)):
 ax.scatter(Treinonly['prev_location'].iloc[i].x,Treinonly['prev_location'].iloc[i].y, s = 3, c = 'red')
plt.show()












