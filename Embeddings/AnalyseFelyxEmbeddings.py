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

model_name = 'All_balanced_deepPCinfo'
APC = [str(PC) for PC in list(gpd.read_file('PublicGeoJsons/AmsPCs.json')['Postcode4'].unique())]
combined = pd.read_pickle(os.path.join('Embeddings','models', model_name, 'predictions'))
Odin = combined[combined.train == 1]
Felyx = combined[combined.train == 0]
Extrainfo = pd.read_pickle('Odin/OdinModellingData/Odin2019All')[list(set(pd.read_pickle('Odin/OdinModellingData/Odin2019All').columns).difference(set(Odin.columns)))]
OdinExtra = Odin.join(Extrainfo)


embFel = Felyx.sample(10000)
embeddings = np.stack(embFel[['Emb' + str(x) for x in range(3)]].values)

from scipy.spatial.distance import pdist, squareform
# Compute cosine similarity matrix
similarity_matrix = distance.euclidean(embeddings)
distances = pdist(embeddings, metric='euclidean')
dist_matrix = squareform(distances)

dist_df = pd.DataFrame(dist_matrix, index=embFel.index, columns=embFel.index)
most_similar_to_cat1 = dist_df[1665].sort_values(ascending=False)

show = pd.concat([embFel.join(most_similar_to_cat1.iloc[:10], how = 'inner'),
embFel.join(most_similar_to_cat1.iloc[-10:], how = 'inner')])
plotshow(show, centers = False)


gr = Felyx.drop(['khvm', 'choice', 'weekdag'], axis = 1 ).groupby('pred').mean()

def readyshow(show, wrong = False, Ams = False, reduce = False, s = 0):
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
def centerdicmake(df):
    centerdict = dict()
    for i in df.choice.unique():
        OO = df[df['choice'] == i]
        points = OO[['Emb0','Emb1', 'Emb2']].values
        centerdict[i] = [statistics.mean(i) for i in zip(*points)]
    return centerdict
def plotshow(show, colorcol = 'choice', centers = True):
    colormap = colormaps.get_cmap('tab20')#, len(show[colorcol].unique()))
    color_dict = {category: colormap(i) for i, category in enumerate(show[colorcol].unique())}
    marker_dict = {1: 'x', 0:'o'}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in show.train.unique():
        showt = show[show.train == i]
        if type(show[colorcol].iloc[0]) == str:
            ax.scatter(showt['Emb0'], showt['Emb1'], showt['Emb2'], c=showt[colorcol].map(color_dict), marker=marker_dict[i])
            for category, color in color_dict.items():
                ax.plot([], [], 'o', color=color, label=category)
            ax.legend()
        else:
            p = ax.scatter(showt['Emb0'], showt['Emb1'], showt['Emb2'], c=showt[colorcol],marker=marker_dict[i])
            plt.colorbar(p)

    if centers == True:
        centerdict = centerdicmake(show)
        for key in centerdict.keys():
            if key in show[colorcol].unique():
                ax.scatter(*centerdict[key], s = 200, c = color_dict[key])#, marker = key[0].lower())

    ax.set_xlabel('Emb0')
    ax.set_ylabel('Emb1')
    ax.set_zlabel('Emb2')
    ax.set_title(colorcol)

    plt.show()

plotshow(readyshow(Odin, reduce = True, wrong = True, s = 100), 'pred')
plotshow(readyshow(Felyx, s = 150))
plotshow(readyshow(Odin, reduce = True, wrong = True, s = 500, Ams =True), 'choice')
plotshow(readyshow(Odin, reduce = True, wrong = True, s = 500, Ams =True), 'pred')
plotshow(evensampletoshow(20), 'choice')

plotshow(readyshow(OdinExtra, reduce = False, wrong = True, s = 500), 'doel')
plotshow(readyshow(OdinExtra, reduce = False, wrong = True, s = 500), 'ovstkaart')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'herkomst')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'prov')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'leeftijd')

for i in Extrainfo.columns:
    print(i, type(Extrainfo[i].iloc[0]))


# Extrainfo['leeftijd'] = Extrainfo['leeftijd'].astype(int)
corr = OdinExtra.drop(['khvm', 'choice', 'weekdag', 'pred'], axis = 1 ).corr()

def pcgroupbyembeds(PC, Ams = True):
    import geopandas as gpd
    vertpcdf = combined[['Emb0', 'Emb1', 'Emb2', PC]].groupby(PC).mean()
    if Ams == True:
        vertpcdf = vertpcdf[(vertpcdf.index.isin(APC))]
    return vertpcdf

def showPCsinEmbedsace(s = 30, both = True, AmsV = True, AmsA = True):
    vertpcdf2 = pcgroupbyembeds('vertpc', AmsV)
    vertpcdf2 = vertpcdf2.sample(min(s, len(vertpcdf2)))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertpcdf2['Emb0'], vertpcdf2['Emb1'], vertpcdf2['Emb2'])
    for i in vertpcdf2.index:
        ax.text(vertpcdf2['Emb0'][i], vertpcdf2['Emb1'][i], vertpcdf2['Emb2'][i], str(i), color='blue')

    if both == True:
        vertpcdf2 = pcgroupbyembeds('aankpc', AmsA)
        vertpcdf2 = vertpcdf2.sample(min(s, len(vertpcdf2)))
        ax.scatter(vertpcdf2['Emb0'], vertpcdf2['Emb1'], vertpcdf2['Emb2'])

        for i in vertpcdf2.index:
            ax.text(vertpcdf2['Emb0'][i], vertpcdf2['Emb1'][i], vertpcdf2['Emb2'][i], str(i), color='red')


    ax.set_xlabel('Emb0')
    ax.set_ylabel('Emb1')
    ax.set_zlabel('Emb2')
    plt.show()
    return
showPCsinEmbedsace(AmsV = True)

def analysepc2D(PC, s, Ams  =True):
    with open(os.path.join('Embeddings', 'models', model_name, 'embedding_dictionary'), 'rb') as f:
        categ_dictionary = pickle.load(f)
    CEdf = categ_dictionary[PC]
    if Ams == True:
        CEdf = CEdf[CEdf.index.isin(APC)]
    redu = PCA(n_components=2).fit_transform(CEdf.values)

    PCDF = pd.DataFrame(data = redu, index = CEdf.index, columns = ['pc' + str(i) for i in range(2)])


    CEdfrs = PCDF.sample(min(s, len(PCDF)))
    p1 = sns.scatterplot(data = CEdfrs, x = 'pc0', y = 'pc1')
    for line in CEdfrs.index:
         p1.text(CEdfrs.pc0[line]+0.01, CEdfrs.pc1[line],
         str(line), horizontalalignment='left',
         size='medium', color='black', weight='light')

    plt.show()

    PCEmb = combined[['Emb0', 'Emb1', 'Emb2', 'vertpc']].groupby('vertpc').mean()
    PCEmb  = PCEmb[(PCEmb.index.isin(APC))]
    corr = PCEmb.join(PCDF).corr()

    return corr
hi =analysepc2D('vertpc', 50, Ams = False)








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
