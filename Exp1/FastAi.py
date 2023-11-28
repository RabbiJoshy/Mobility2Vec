from ModellingUtilities import *
from fastai.tabular.all import *
from Exp1.X import collate, collatebinary, age_cat
from collections import Counter

def oversample_df(df):
    max_size = df['khvm'].value_counts().max()
    return pd.concat([df[df['khvm'] == cls].sample(max_size, replace=True) for cls in df['khvm'].unique()])

def prep(type, balanced, binary):
    target = 'khvm'
    otp_cols = ['pt_duration', 'pt_distance', 'car_duration', 'car_distance', 'bike_duration', 'bike_distance',
                'walk_distance']  # 'distance_moved'
    cost_cols = ['AVG_CHARGE', 'car_cost', 'pt_cost_GVB']  # 'felyx_cost', 'pt_cost'
    meteo_cols = ['windspeed', 'temp', 'precip', 'precipcover']
    continuous_cols = otp_cols + cost_cols + meteo_cols
    category_cols = ['hour', 'CanFelyx', 'weekdag']  # ,#'doel''kmotiefv', 'hhgestinkg'
    category_cols += ['leeftijd_cat']  # , 'doel']#,'ovstkaart' , 'hhauto'] #geslacht', 'herkomst', 'opleiding'

    print(type, balanced, binary)
    O = pd.read_pickle(os.path.join('OdinData', 'OdinWrangled', 'Odin2018-2021' + type))
    O = age_cat(O)
    O['hour'] = O.prev_time.dt.hour
    F = pd.read_pickle(os.path.join('FelyxData', 'FelyxModellingData', 'felyxotpAADO'))
    F['hour'] = F.prev_time.dt.hour
    O = O[continuous_cols + category_cols + [target]]
    # O[meteo_cols] = np.round(O[meteo_cols] / 5) * 5

    if binary == False:
        O_collated = collate(O)
    else:
        O_collated = collatebinary(O)

    F['leeftijd'] = 27
    F = age_cat(F)
    F['geslacht'] = 'Man'
    F['opleiding'] = 'Hoger beroepsonderwijs, universiteit'
    F['herkomst'] = 'Nederlandse achtergrond'

    return O_collated, F, continuous_cols, category_cols, target

for type in ['UrbanAnd']:
    for balanced in [True]:
        for binary in [False]:

            O, F, continuous_cols, category_cols, target = prep(type, balanced, binary)

            if balanced == True:
                O = oversample_df(O)

            procs = [Categorify, FillMissing, Normalize]
            splits = RandomSplitter(valid_pct=0.2)(range_of(O))
            to = TabularPandas(O, procs=procs, cat_names=category_cols,
                               cont_names=continuous_cols, y_names=target,
                               splits = splits, y_block = CategoryBlock())
            dls = to.dataloaders(bs=64, splits=splits)
            # learn = tabular_learner(dls, layers=layers, metrics=accuracy)
            learn = tabular_learner(dls, metrics=accuracy)

            lr = learn.lr_find()

            learn.fit_one_cycle(1, lr)
            interp = ClassificationInterpretation.from_learner(learn)
            interp
            interp.print_classification_report()


            learn.predict(F.iloc[0])

            test_cols = [col for col in O.columns if col != target]
            test_df = F[test_cols]

            dl = learn.dls.test_dl(test_df)
            #
            preds, _ = learn.get_preds(dl=dl)#, with_decoded=True)
            preds = preds.argmax(dim=-1)
            preds_trans = [dl.vocab[pred] for pred in preds]
            modalsplit = Counter(preds_trans)
            splits = []
            modes = ['Personenauto - bestuurder', 'Fiets', 'Te voet', 'Bus/tram/metro']#, 'Personenauto - passagier', 'Trein']
            #
            for mode in modes:
                splits.append(str(round((modalsplit[mode]/len(preds_trans)* 100),1)))
            print('NN', '&', balanced, '&', type, '&', round(88,2),'&','/'.join(splits))
            # test_df.groupby('pred').count()


