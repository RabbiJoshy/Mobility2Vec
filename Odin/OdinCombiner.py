import pandas as pd
O1 = pd.read_csv('Odin/OdinData/odin-2018-2019-v1.0.0.csv')
O2 = pd.read_csv('Odin/OdinData/Odin2017.csv')

O1 = O1.replace({"khvm": {'3':'4'}})
# sdf = sdf.replace({"khvm": {'2':'1'}})
O1 = O1[O1.khvm != '7']


O1.columns
O2.columns
di = {'choice':'khvm',
        'bike_dur': 't_cycle',
    'bike_dist':  'dist_cycle',
       'car_dur': 't_car',
       'car_dist': 'dist_car',
       'pt_dur': 't_transit',
       'pt_dist': 'dist_transit',
      'walk_dur':'t_walk',
      'walk_dist': 'dist_walk'
}
mapping = {v: k for k, v in di.items()}
O2 = O2.rename(columns=mapping)

o1set = set(O1.columns)
o2set = set(O2.columns)
common = list(o1set.intersection(o2set))

combined = pd.concat([O1[common], O2[common]])

combined.to_pickle('Odin/OdinData/combined')

df = combined
df = df[((df['aankpc'].between(1011, 1109)) |
(df['aankpc'].between(1381, 1384))) & ((df['vertpc'].between(1011, 1109)) |
(df['vertpc'].between(1381, 1384)))]
