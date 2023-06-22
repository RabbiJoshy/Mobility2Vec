import pandas as pd
cols = ['vertpc', 'aankpc'] + ['geslacht', 'leeftijd', 'herkomst','doel', 'kmotiefv'] + ['hvm' ,'khvm'] + \
       ['vertmin','dag','jaar','maand', 'vertuur'] + ['verplid'] + ['weekdag', 'feestdag', 'oprijbewijsau'] + \
       ['opleiding','hhgestinkg', 'oprijbewijsmo', 'hhauto', 'oprijbewijsbr']
from datetime import datetime
o2019 = pd.read_pickle('/Users/joshuathomas/Desktop/Odin2019Full')
o2018 = pd.read_pickle('/Users/joshuathomas/Desktop/Odin2018Full')
o2020 = pd.read_pickle('/Users/joshuathomas/Desktop/Odin2020Full')
o2021 = pd.read_pickle('/Users/joshuathomas/Desktop/Odin2021Full')
otp = pd.read_pickle('/Users/joshuathomas/Desktop/otp_allyears').drop('opid', axis = 1).set_index('verplid')

o2021.columns

# set(o2020.columns).difference(set(o2018.columns))
combined = pd.concat([o2020[o2018.columns], o2019[o2018.columns], o2018, o2021])[cols]
combined['datetime'] = pd.to_datetime(dict(day = combined.dag, year = combined.jaar,
                                           month = combined.maand, hour = combined.vertuur, minute = combined.vertmin))
# combined= combined.dropna(subset=['verplid'])
combined = combined.set_index('verplid')

combined = combined[combined.index.notnull()]
combined.index = combined.index.astype(int)
combined = combined[~combined.index.duplicated()]#.reset_index()

# inv = combined[['oprijbewijsmo', 'oprijbewijsau', 'oprijbewijsbr', 'khvm']]
# inv['has_license'] = inv['oprijbewijsmo'] | inv['oprijbewijsau']| inv['oprijbewijsbr']
# inv.groupby(['khvm', 'has_license']).count()

combotp = combined.join(otp)
otp.index
combined.index
combotp.walk_distance.isna().sum()

combotp.index
for col in ['vertpc', 'aankpc', 'vertmin' ,'vertuur', 'doel', 'kmotiefv', 'hvm', 'khvm']:
       combotp = combotp.dropna(subset=[col])
       combotp[col] = combotp[col].astype(int)

combotp = combotp[combotp.aankpc != combotp.vertpc]
combotp = combotp[combotp.aankpc != 0]
combotp = combotp[combotp.vertpc != 0]


def getmeteo():
       meteo21 = pd.read_csv('Weather/Daily/Amsterdam2021Daily.csv')
       meteo20 = pd.read_csv('Weather/Daily/Amsterdam2020Daily.csv')
       meteo19 = pd.read_csv('Weather/Daily/Amsterdam2019Daily.csv')
       meteo18 = pd.read_csv('Weather/Daily/Amsterdam2018Daily.csv')
       meteo = pd.concat([meteo18, meteo19, meteo20,meteo21]).drop_duplicates(subset = 'datetime')
       meteo.datetime = pd.to_datetime(meteo.datetime)
       meteo['jaar'] = meteo['datetime'].dt.year
       meteo['maand'] = meteo['datetime'].dt.month
       meteo['dag'] = meteo['datetime'].dt.day
       return meteo
def joinmeteo(df, relcols = ['windspeed', 'temp', 'feelslike', 'precip', 'precipcover']):
       meteo = getmeteo()
       df = df.reset_index()
       meteojoined = df.merge(meteo[['dag','jaar','maand'] + relcols], on = ['dag','jaar','maand'])
       meteojoined = meteojoined.set_index('verplid')
       return meteojoined

meteojoined = joinmeteo(combotp)
meteojoined[['dag','jaar','maand']] = meteojoined[['dag','jaar','maand']].astype(str)
meteojoined.to_pickle('Odin/odin2018-2021')

meteojoined.dropna()
licenses = combined.groupby(['oprijbewijsmo', 'oprijbewijsau']).count()
licenses = combined.groupby(['oprijbewijsau']).count()



pclist = ['aank Man', 'aank Huurwoning', 'aank Age0', 'aank Age1', 'aank hh0',
       'aank hh1', 'aank build_age0', 'aank build_age1', 'aank Immigration0',
       'aank Immigration1', 'aank size0', 'aank size1', 'vert Man',
       'vert Huurwoning', 'vert Age0', 'vert Age1', 'vert hh0', 'vert hh1',
       'vert build_age0', 'vert build_age1', 'vert Immigration0',
       'vert Immigration1', 'vert size0', 'vert size1']
# pccols = ['aank ' + x for x in pclist] + ['vert ' + x for x in pclist]

import json
FeaturesDict = dict()
# FeaturesDict['OTP'] = ['bike_dur', 'bike_dist', 'car_dur', 'car_dist', 'pt_dur', 'pt_dist', 'walk_dur', 'walk_dist']
FeaturesDict['OTP'] = ['walk_distance', 'bike_distance', 'car_distance', 'pt_distance', 'walk_duration', 'bike_duration', 'car_duration', 'pt_duration']
# FeaturesDict['FelyxKnown'] = ['choice_dur', 'choice_dist', 'oprijbewijsmo']
FeaturesDict['FelyxKnown'] = ['odin_duration', 'odin_distance', 'CanFelyx']
FeaturesDict['License'] = ['oprijbewijsmo', 'oprijbewijsau']
FeaturesDict['Comparison'] = ['geslacht', 'leeftijd', 'herkomst', 'opleiding','doel', 'kmotiefv','hhgestinkg', 'hhauto', 'prev_time']
FeaturesDict['Location'] = ['vertpc', 'aankpc']
FeaturesDict['Time'] = ['sin_time', 'cos_time', 'weekdag', 'feestdag', 'jaar', 'maand']
FeaturesDict['Targets'] = ['hvm' ,'khvm']
FeaturesDict['PCInfo'] = pclist
FeaturesDict['Weather'] = ['windspeed', 'temp', 'feelslike', 'precip', 'precipcover']
with open('Features_Dictionary', 'w') as outfile:
    json.dump(FeaturesDict, outfile)

