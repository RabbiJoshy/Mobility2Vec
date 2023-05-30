import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import geopandas as gpd
import pickle
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'All_balanced_deep'
APC = list(gpd.read_file('PublicGeoJsons/AmsPCs.json')['Postcode4'].unique())
combined = pd.read_pickle(os.path.join('Embeddings','models', model_name, 'predictions'))

odin = combined[combined.train == 1]
felyx = combined[combined.train == 0]
odinams = odin[odin.aankpc.isin(APC)]
# outliers = show[(show.Emb1 > 0) & (show.Emb0 > -1)]
# bigoutliers = show[(show.Emb1 > 1) & (show.Emb0 > 0)]
# plotshow(bigoutliers)
# inliers = show[~(show.index.isin(bigoutliers.index))]
# plotshow(inliers)

def reduceshow(show):
    reduced = show.copy()
    # redu = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(reduced[['Emb'+str(x) for x in range(3)]].values)
    redu = PCA(n_components=3).fit_transform(reduced[['Emb' + str(x) for x in range(3)]].values)
    reduced[['Emb' + str(x) for x in range(3)]] = redu

    return reduced

def plotshow(show):
    colormap = cm.get_cmap('Accent', len(show['choice'].unique()))
    color_dict = {category: colormap(i) for i, category in enumerate(show['choice'].unique())}
    marker_dict = {1: 'x', 0:'o'}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in show.train.unique():
        showt = show[show.train == i]
        ax.scatter(showt['Emb0'], showt['Emb1'], showt['Emb2'], c=showt.choice.map(color_dict), marker=marker_dict[i])
    ax.set_xlabel('Emb0')
    ax.set_ylabel('Emb1')
    ax.set_zlabel('Emb2')
    for category, color in color_dict.items():
        ax.plot([], [], 'o', color=color, label=category)
    ax.legend()
    # plt.colorbar()
    plt.show()

reducedshow = reduceshow(odin)
# reducedshow[reducedshow.choice != '']
plotshow(reducedshow.sample(1000))
plotshow(felyx.sample(100))
plotshow(odinams.sample(100))



def sampletoshow(n):
    show = pd.DataFrame()
    for i in combined.choice.unique():
        for j in combined.train.unique():
            print(i,j)
        # print(combined[combined.choice == i])
            boys = combined[(combined.choice == i) & (combined.train == j)]
            print(len(boys))
            show = pd.concat([show, boys.sample(min(n, len(boys)))])
    return show
show = sampletoshow(20)

plotshow(show)
corr = combined.select_dtypes(include=np.number).corr()



def viewtrainjourneys(col, plot = False):
    # trains = combined[combined.choice == 'Trein']
    trains = pd.read_pickle('Odin/Odin2019Ams')
    trains = trains[trains.khvm == 'Trein']
    tra = trains.groupby(col).count()['khvm']

    g = gpd.read_file('PublicGeoJsons/AmsPCs.json')
    g = g.set_index('Postcode4')
    # g['Postcode4'] = g['Postcode4'].astype(str)

    # bijlmerweesp = [1101, 1102, 1103, 1104, 1105, 1381, 1382, 1383, 1384]
    # feasible = t[(~t.vertpc.isin(bijlmerweesp)) & (~t.aankpc.isin(bijlmerweesp))]
    # feasible.aankpc.unique()

    if plot == True:
        fig, ax = plt.subplots()
        g[g.index.isin(tra.index)].join(tra).plot(ax=ax, column = 'khvm', legend = True)
        g[~g.index.isin(tra)].join(tra).plot(ax=ax, facecolor = 'None')
    return g, trains
g, t = viewtrainjourneys('aankpc')

bijlmerweesp = [1101, 1102, 1103, 1104,1105, 1381, 1382, 1383, 1384]
feasible = t[(~t.vertpc.isin(bijlmerweesp)) & (~t.aankpc.isin(bijlmerweesp)) ]
feas = feasible.aankpc.unique()
g[g.index.isin(feas)].plot()

def pcgroupbyembeds(PC):
    import geopandas as gpd
    APC = list(gpd.read_file('PublicGeoJsons/AmsPCs.json')['Postcode4'].unique())
    vertpcdf = combined[['Emb0', 'Emb1', 'Emb2', PC]].groupby(PC).mean()
    vertpcdf2 = vertpcdf[(vertpcdf.index.isin(APC))]
    return vertpcdf2

def showPCsinEmbedsace(PC):
    vertpcdf2 = pcgroupbyembeds(PC)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertpcdf2['Emb0'], vertpcdf2['Emb1'], vertpcdf2['Emb2'])
    ax.set_xlabel('Emb0')
    ax.set_ylabel('Emb1')
    ax.set_zlabel('Emb2')
    for i in vertpcdf2.index:
        ax.text(vertpcdf2['Emb0'][i], vertpcdf2['Emb1'][i], vertpcdf2['Emb2'][i], str(i))
    # ax.text(0, 0, 0, "red", color='red')
    plt.show()
    return
showPCsinEmbedsace('vertpc')

def analysepcreductions(PC):
    APC = list(gpd.read_file('PublicGeoJsons/AmsPCs.json')['Postcode4'].unique())
    with open(os.path.join('Embeddings', 'models', model_name, 'embedding_dictionary'), 'rb') as f:
        categ_dictionary = pickle.load(f)
    CEdf = categ_dictionary[PC]
    amsdf = CEdf[(CEdf.index.isin(APC)) &(CEdf.index.isin(APC))]
    redu = PCA(n_components=2).fit_transform(amsdf.values)

    PCDF = pd.DataFrame(data = redu, index = amsdf.index, columns = ['pc' + str(i) for i in range(2)])


    CEdfrs = PCDF.sample(50)
    p1 = sns.scatterplot(data = CEdfrs, x = 'pc0', y = 'pc1')
    for line in CEdfrs.index:
         p1.text(CEdfrs.pc0[line]+0.01, CEdfrs.pc1[line],
         str(line), horizontalalignment='left',
         size='medium', color='black', weight='light')

    plt.show()


    vertpcdf = combined[['Emb0', 'Emb1', 'Emb2', 'vertpc']].groupby('vertpc').mean()
    vertpcdf2  = vertpcdf[(vertpcdf.index.isin(APC))]

    vertpcdf2.join(PCDF).corr()

    return
analysepcreductions('vertpc')


