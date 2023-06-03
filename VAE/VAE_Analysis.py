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
encoder = LabelEncoder()
APC = gpd.read_file('PublicGeoJsons/AmsPCs.json')
VAE_dimensionality = 10
n_components=3
df = pd.read_pickle('VAE/WithoutKHVM/Odin/E10')
odin2019 = pd.read_pickle('Odin/OdinModellingData/Odin2019All')
df['khvm'] = odin2019['khvm']




#TODO - fix odin columns to be categorical, fix the felyxotp choice dur thing before modelling, fix

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
felyx, _ = ReduceDF(felyx, fit = pca)

def plotodin(Odin, colorcol='choice', centers=True):
 colormap = colormaps.get_cmap('tab20')  # , len(show[colorcol].unique()))
 color_dict = {category: colormap(i) for i, category in enumerate(Odin[colorcol].unique())}

 fig = plt.figure(figsize=(8, 6))
 ax = fig.add_subplot(111, projection='3d')
 ax.scatter(*[Odin['PC' +str(x)] for x in range(3)], c=Odin[colorcol].map(color_dict))


 for category, color in color_dict.items():
  ax.plot([], [], 'o', color=color, label=category)
  ax.legend()

 ax.set_xlabel('PC0')
 ax.set_ylabel('PC1')
 ax.set_zlabel('PC2')
 ax.set_title(colorcol)

 plt.show()
# plotshow(df.sample(3000), colorcol = 'khvm')
plotshow(amsdf.sample(200), colorcol = 'khvm')

def xgc(df, samples =0, reduce = 0, target = 'khvm'):
 if samples > 0:
  df = df.sample(samples, random_state=42)
 df[target] = encoder.fit_transform(df[target])

 model = xgb.XGBClassifier()
 X = df[['emb' + str(x) for x in range(10)]]
 if reduce > 0:
  X = df[['PC' + str(x) for x in range(3)]]
 y = df[target]

 X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.33, random_state=42)

 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)

 X_test['predictions'] = encoder.inverse_transform(y_pred)
 X_test['khvm'] = encoder.inverse_transform(y_test)

 print(accuracy_score(y_test, y_pred))
 cm = confusion_matrix(y_test, y_pred)
 types = encoder.inverse_transform(list(range(len(df[target].unique()))))
 print(types)
 cmdf = pd.DataFrame(cm, index=types, columns=types)
 print(classification_report(y_test, y_pred, target_names= types))

 return X_test, model, encoder, cmdf

TestDF, classifier, encoder, cm = xgc(df, 100000)
TestDF =TestDF.join(df[['aankpc', 'vertpc']])
amsdf = TestDF[TestDF.aankpc.isin([str(x) for x in APC.Postcode4.unique()]) & TestDF.vertpc.isin([str(x) for x in APC.Postcode4.unique()])]
amsdf, _ = ReduceDF(amsdf, fit = pca)
amsclf, amsenc, amscm = xgc(amsdf)


felyx = pd.read_pickle('VAE/WithoutKHVM/Felyx/E10')
preds = classifier.predict(felyx[['emb' + str(x) for x in range(10)]])
felyx['khvm'] = encoder.inverse_transform(preds)
felyx['khvm'].value_counts()

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
plotboth(felyx.sample(1000), amsdf.sample(1000))
plotboth(felyx[felyx.khvm ==  'Te voet'] ,amsdf[amsdf.khvm ==  'Te voet'] )


odin2019.doel = odin2019.doel.astype(str)
classifier, encoder, cm = xgc(df.join(odin2019.doel), 5000, target = 'doel')







Treinonly = Fel[Fel.prediction == 'Trein'][['aankpc', 'vertpc', 'weekdag', 'sin_time']]
Treinonly.aankpc.unique()

otpfelyx = pd.read_pickle('FelyxData/FelyxModellingData/FelyxOTP')

Treinonly = Treinonly.join(otpfelyx[['lat', 'lon', 'prev_location']])

fig, ax = plt.subplots()
APC.plot(ax = ax, facecolor = 'None')
for i in range(len(Treinonly)):
 ax.scatter(Treinonly['prev_location'].iloc[i].x,Treinonly['prev_location'].iloc[i].y, s = 3, c = 'red')
plt.show()












