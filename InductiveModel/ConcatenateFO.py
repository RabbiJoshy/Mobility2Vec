import pandas as pd
import os
type = 'Ams'
O = pd.read_pickle('Odin/OdinWrangled/Odin2018-2021' + type)
Fel = pd.read_pickle('FelyxData/FelyxModellingData/felyxotpAmsterdam')
O.khvm.value_counts()[0]
Fel = Fel.sample(min(len(Fel), O.khvm.value_counts()[0]))
Fel['hvm'] = len(Fel) * ['Felyx'] #TODO move to wrangler
Fel['khvm'] = len(Fel) * ['Felyx']

# def getmeteo():
#     meteo19 = pd.read_csv('Weather/Daily/Amsterdam2019Daily.csv')
#     meteo18 = pd.read_csv('Weather/Daily/Amsterdam2018Daily.csv')
#     meteo23 = pd.read_csv('Weather/Daily/Amsterdam2023Daily.csv')
#     meteo = pd.concat([meteo18, meteo19, meteo23]).drop_duplicates(subset='datetime')
#     meteo.datetime = pd.to_datetime(meteo.datetime)
#     meteo['jaar'] = meteo['datetime'].dt.year
#     meteo['maand'] = meteo['datetime'].dt.month
#     meteo['dag'] = meteo['datetime'].dt.day
#     return meteo
#
#
# def joinmeteo(df, relcols=['windspeed', 'temp', 'feelslike', 'description']):
#     meteo = getmeteo()
#     meteo['date'] = meteo['datetime'].dt.date
#     meteojoined = df.merge(meteo[['date'] + relcols], on=['date'])
#     meteojoined = meteojoined.drop('date', axis = 1)
#
#     meteojoined = meteojoined.set_index('tripid')
#     return meteojoined
#
# meteo = getmeteo()
# Fel['date'] = Fel['prev_time'].dt.date
#
# FelMeteo = joinmeteo(Fel)


shared = list(set(O.columns).intersection(set(Fel.columns)))
conc = pd.concat([O[shared], Fel[shared]])
conc = conc.replace({'feestdag': {'Ja':1, 'Nee':0}})
conc.to_pickle(os.path.join('InductiveModel', type, 'Raw'))
conc.feestdag.unique()