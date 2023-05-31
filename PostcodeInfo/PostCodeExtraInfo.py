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

first_col = df.iloc[:, 0]  # Get the first column
df_percent = df.iloc[:, 1:].div(first_col, axis=0).mul(100)

# Concatenate the first column with the percentage columns
df_result = pd.concat([first_col, df_percent], axis=1)
df_result = df_result.join(density)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Apply Min-Max scaling to all columns
df_scaled = pd.DataFrame(scaler.fit_transform(df_result), columns=df_result.columns, index = df_result.index)



# cols = list(df_result.columns)
# df_result[cols] = df_result[cols] / df_result[cols].sum()

df_scaled.to_pickle('PostcodeInfo/PCData')
df_reduced.to_pickle('PostcodeInfo/PCDataReduced')

df_reduced = df_scaled.copy()
redu = PCA(n_components=5).fit_transform(df_reduced.values)
df_reduced[['Emb' + str(x) for x in range(5)]] = redu
df_reduced = df_reduced.join(urbanity)
corr = df_reduced.corr()


df_show = df_reduced[df_reduced.urbanity.isin([2,4])]

ax = sns.scatterplot(data = df_show, x = 'Emb0', y= 'Emb1', hue = 'urbanity', s = 7) #'Unnamed: 37')'Totaal')
for i, txt in enumerate(df_show.index):
    ax.text(df_show['Emb0'][i], df_show['Emb1'][i], str(txt), ha='center', va='bottom', fontsize = 4)
