import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colormaps
import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import statistics
PCs = gpd.read_file('PublicGeoJsons/AmsterdamPC4.geojson')
APC = [str(PC) for PC in list(PCs['Postcode4'].unique())]

def evensampletoshow(n):
    show = pd.DataFrame()
    for i in combined.choice.unique():
        for j in combined.train.unique():
            print(i,j)
        # print(combined[combined.choice == i])
            boys = combined[(combined.choice == i) & (combined.train == j)]
            print(len(boys))
            show = pd.concat([show, boys.sample(min(n, len(boys)))])
    return show

def plotboth(felyx, Odin, s, colorcol = 'choice', centers = True):

 """UNREDUCED"""
 felyx = felyx
 Odin = Odin
 Axes = 'Emb'

 colormap = colormaps.get_cmap('tab20')  # , len(show[colorcol].unique()))
 types = list(set(felyx[colorcol].unique()).union(set(Odin[colorcol].unique())))
 print(types)
 color_dict = {category: colormap(i) for i, category in enumerate(types)}


 fig = plt.figure(figsize=(8, 6))
 ax = fig.add_subplot(111, projection='3d')
 ax.scatter(*[Odin[Axes + str(x)] for x in range(3)], c=Odin[colorcol].map(color_dict), marker="$0$")
 ax.scatter(*[felyx[Axes + str(x)] for x in range(3)], c=felyx[colorcol].map(color_dict), marker="$f$")

 for category, color in color_dict.items():
  ax.plot([], [], 'o', color=color, label=category)
  ax.legend()

 ax.set_xlabel('Emb0')
 ax.set_ylabel('Emb1')
 ax.set_zlabel('Emb2')
 ax.set_title(colorcol)

 if centers == True:
     combined = pd.concat([felyx, Odin])
     centerdict = centerdicmake(combined)
     for key in centerdict.keys():
         if key in combined[colorcol].unique():
             ax.scatter(*centerdict[key], s=250, c=color_dict[key])

 return

def centerdicmake(df):
    centerdict = dict()
    for i in df.choice.unique():
        OO = df[df['choice'] == i]
        points = OO[['Emb0','Emb1', 'Emb2']].values
        centerdict[i] = [statistics.mean(i) for i in zip(*points)]
    return centerdict

def filter(show, s = 0, wrong = False, Ams = False, reduce = False):
    if wrong == True:
        show = show[show.choice != show.pred]
    if Ams == True:
        show = show[show.aankpc.isin(APC)]
    if reduce == True:
        reduced = show.copy()
        # redu = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(reduced[['Emb'+str(x) for x in range(3)]].values)
        redu = PCA(n_components=3).fit_transform(reduced[['Emb' + str(x) for x in range(3)]].values)
        show[['Emb' + str(x) for x in range(3)]] = redu
    if s > 0:
        show = show.sample(s)

    return show


def plotjourneys(col, samples, df):
    best = df[df.choice == col]
    fig, ax = plt.subplots()
    PCs.plot(ax = ax, facecolor = 'None', linewidth = 0.1)
    ax.set_aspect('equal')
    trein = best.sample(samples)
    for index, row in trein.iterrows():
        x = row['geometry'].x
        y = row['geometry'].y
        prev_x = row['prev_location'].x
        prev_y = row['prev_location'].y
        dx = x - prev_x
        dy = y - prev_y
        ax.arrow(prev_x, prev_y, dx, dy, width = 0.000005, head_width=0.001, head_length=0.005, fc='green', ec='blue')
        ax.text(prev_x, prev_y, "{}:{}".format(row['prev_time'].hour, row['prev_time'].minute), c = 'b', fontweight = 'bold')
    return


def plotjourneys2(df, background, startPC, endPC, modechoice = 'Personenauto - bestuurder', samples = 0, choicecol = 'khvm', time = False):
    best = df[df[choicecol] == modechoice]
    fig, ax = plt.subplots()
    for i in background:
        i.plot(ax = ax, facecolor = 'None', linewidth = 0.1)
    asa.plot(ax=ax, linewidth=1, alpha = 0.5)
    # ax.set_aspect('equal')
    if samples > 0:
        best = best.sample(samples)
    for index, row in best.iterrows():
        x = AP.loc[row[startPC]].centroid.x
        y = AP.loc[row[startPC]].centroid.y
        prev_x = AP.loc[row[endPC]].centroid.x
        prev_y = AP.loc[row[endPC]].centroid.y
        dx = x - prev_x
        dy = y - prev_y
        ax.arrow(prev_x, prev_y, dx, dy, width = 0.00000005, head_width=0.001, head_length=0.005, fc='green', ec='blue')
        if time == True:
            ax.text(prev_x, prev_y, "{}:{}".format(row['prev_time'].hour, row['prev_time'].minute), c = 'b', fontweight = 'bold')
    return
# plotjourneys(outside, [AP, asa], 'vertpc', 'aankpc', samples =150)