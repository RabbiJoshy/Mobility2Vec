import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('PostcodeInfo/2023-CBS_pc4_2022_v1/PC4.csv')
df = df.iloc[1:, :]
df = df.set_index('Postcode-4')
df.index = df.index.astype(str)
df = df.rename(columns = {'Unnamed: 36': 'density'})
df = df.rename(columns = {'Unnamed: 37': 'urbanity'})
df = df.astype(float)
df.index = df.index.astype(str)
urbanity = df['urbanity']
density = df['density']
df = df.iloc[:, :-2]
df = df.replace(-99997, 2)
df.columns

#Gender
df = df.drop('Vrouw', axis =1)
df.Man = df.Man.div(df.Totaal, axis =0).mul(100)

age = ['tot 15 jaar', '15 tot 25 jaar', '25 tot 45 jaar', '45 tot 65 jaar', '65 jaar en ouder']
df[age] = df[age].div(df.Totaal, axis =0).mul(100)


# pca =  PCA(n_components=2).fit(df[age].values)
# redu = pca.transform(df[age].values)
# df[['Age' + str(x) for x in range(2)]] = redu
# df = df.drop(age, axis = 1)


#Househoulds
hh = ['Eenpersoons', 'Meerpersoons \nzonder kinderen', 'Eenouder', 'Tweeouder', 'Huishoudgrootte']
df[hh] = df[hh].div(df['Totaal.1'], axis =0).mul(100)
# pca =  PCA(n_components=2).fit(df[hh].values)
# redu = pca.transform(df[hh].values)
# df[['hh' + str(x) for x in range(2)]] = redu
# df = df.drop(hh, axis = 1)

build_age = ['voor 1945','1945 tot 1965', '1965 tot 1975', '1975 tot 1985', '1985 tot 1995', '1995 tot 2005', '2005 tot 2015', '2015 en later']
df[build_age]  = df[build_age].div(df['Totaal.2'], axis =0).mul(100)

# pca =  PCA(n_components=2).fit(df[build_age].values)
# redu = pca.transform(df[build_age].values)
# df[['build_age' + str(x) for x in range(2)]] = redu
# df = df.drop(build_age, axis = 1)


Im_Stat = ['Geboren in Nederland met een Nederlandse herkomst',
       'Geboren in Nederland met een Europese herkomst (excl. Nederland)',
       'Geboren in Nederland met herkomst buiten Europa',
       'Geboren buiten Nederland met een Europese herkomst (excl. Nederland)',
       'Geboren buiten Nederland met een herkomst buiten Europa']
# pca =  PCA(n_components=2).fit(df[Im_Stat].values)
# redu = pca.transform(df[Im_Stat].values)
# df[['Immigration' + str(x) for x in range(2)]] = redu
# df = df.drop(Im_Stat, axis = 1)
df.columns

df = df.join(density)

df.to_pickle('PostcodeInfo/PC4_Clean')






size = ['Totaal', 'Totaal.1', 'Totaal.2', 'Meergezins', 'density']
pca =  PCA(n_components=2).fit(df[size].values)
redu = pca.transform(df[size].values)
df[['size' + str(x) for x in range(2)]] = redu
df = df.drop(size, axis = 1)

df = df.drop('Koopwoning', axis =1)
df = df.drop(['Huurcoporatie', 'Niet bewoond','WOZ-waarde\nwoning',  'Personen met WW, Bijstand en/of AO uitkering\nBeneden AOW-leeftijd'],axis =1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max scaling to all columns
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index = df.index)



# cols = list(df_result.columns)
# df_result[cols] = df_result[cols] / df_result[cols].sum()

df_scaled.to_pickle('PostcodeInfo/PCData')
df_reduced.to_pickle('PostcodeInfo/PCDataReduced')

pd.read_pickle('PostcodeInfo/PCDataReduced').columns


df_reduced = df_scaled.copy()
pca =  PCA(n_components=4).fit(df_reduced.values)
redu = pca.transform(df_reduced.values)
df_reduced[['Emb' + str(x) for x in range(4)]] = redu
pcadf = pd.DataFrame(pca.components_, columns = df_scaled.columns).T
df_reduced = df_reduced.join(urbanity)
corr = df_reduced.corr()


df_show = df_reduced[df_reduced.urbanity.isin([2,4])]

ax = sns.scatterplot(data = df_show, x = 'Emb0', y= 'Emb1', hue = 'urbanity', s = 7) #'Unnamed: 37')'Totaal')
for i, txt in enumerate(df_show.index):
    ax.text(df_show['Emb0'][i], df_show['Emb1'][i], str(txt), ha='center', va='bottom', fontsize = 4)
