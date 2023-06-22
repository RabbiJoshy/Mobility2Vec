from Embeddings.EmbeddingUtilities import *

import json
file = '/Users/joshuathomas/Desktop/2023-03-01_territories.json'
with open(file, 'r') as f:
  data = json.load(f)[0]
p = data['polygons'][0]['coordinates']
data.keys()

model_name = 'All_notrein_deepPCinfo'
model_name = 'All_balanced_notrein_deepPCinfo'
combined = pd.read_pickle(os.path.join('Embeddings','models', model_name, 'predictions'))
Odin = combined[combined.train == 1]
Felyx = combined[combined.train == 0]
Extrainfo = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021All')[list(set(pd.read_pickle('Odin/OdinWrangled/Odin2018-2021All').columns).difference(set(Odin.columns)))]
OdinExtra = Odin.join(Extrainfo)

# embFel = Felyx.sample(10000)
# embeddings = np.stack(embFel[['Emb' + str(x) for x in range(3)]].values)

plotboth(filter(Felyx, 1000, Ams = True), filter(Odin, 100, Ams = True), 'choice')

e = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAmsterdam')[['prev_location','geometry', 'prev_time']]
d = Felyx.join(e)
plotjourneys('Personenauto - bestuurder', 5, d)

# from scipy.spatial.distance import pdist, squareform
# # Compute cosine similarity matrix
# similarity_matrix = distance.euclidean(embeddings)
# distances = pdist(embeddings, metric='euclidean')
# dist_matrix = squareform(distances)

# dist_df = pd.DataFrame(dist_matrix, index=embFel.index, columns=embFel.index)
# most_similar_to_cat1 = dist_df[1665].sort_values(ascending=False)

# show = pd.concat([embFel.join(most_similar_to_cat1.iloc[:10], how = 'inner'),
# embFel.join(most_similar_to_cat1.iloc[-10:], how = 'inner')])
# plotshow(show, centers = False)
# gr = Felyx.drop(['khvm', 'choice', 'weekdag', 'description'], axis = 1 ).groupby('pred').mean()
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

plotshow(filter(OdinExtra, reduce = False, wrong = True, s = 500), 'doel')
plotshow(readyshow(OdinExtra, reduce = False, wrong = True, s = 500), 'ovstkaart')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'herkomst')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'prov')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'leeftijd')
plotshow(readyshow(OdinExtra, reduce = False, wrong = False, s = 500), 'geslacht')


Extrainfo['leeftijd'] = Extrainfo['leeftijd'].astype(int)
corr = OdinExtra.drop(['doel', 'herkomst', 'hhauto', 'opleiding', 'hhgestinkg', 'hvm',
       'geslacht', 'leeftijd', 'kmotiefv', 'oprijbewijsau', 'weekdag', 'maand', 'jaar', 'oprijbewijsmo', 'feestdag', 'hvm' , 'khvm', 'choice', 'pred'], axis = 1 ).corr()
OdinExtra.columns

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
