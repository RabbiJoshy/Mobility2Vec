import pandas as pd
embeddingsdict = pd.read_pickle('Embeddings/FastAI/embeddingsdict')

data = pd.read_pickle('Embeddings/FakeFelyx/FakeFelyxData')
# data = data[data['aankpc'].str.len() > 0]
#
# data['to_embed'] = data['aankpc'].astype(str) + data['hour'].astype(str)
# data = data.drop(['aankpc', 'hour'], axis = 1)


vertdf = embeddingsdict['vertpc']
vertdf.index = vertdf.index.astype(str)

to_embeddf = embeddingsdict['to_embed']

v = data.set_index('vertpc').join(vertdf).set_index('to_embed')

y = v.join(to_embeddf, rsuffix = '_to_embed').dropna()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
y.columns = y.columns.astype(str)
reduced_features = pca.fit_transform(y[y.columns[2:]])

df_reduced = pd.DataFrame(data=reduced_features, columns=['PC1', 'PC2'])

# Concatenate the reduced features with the original dataframe
df_final = pd.concat([y[y.columns[:2]].reset_index(drop = True), df_reduced], axis=1)


df_final.to_pickle('FakeFelyxDataEmbedded')

