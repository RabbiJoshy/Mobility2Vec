from ModellingUtilities import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
def train_xgboost_weighted(df_train, df_test, category_cols, continuous_cols, target_col):
    xgb_model = xgb.XGBClassifier()

    df_combined = pd.concat([df_train, df_test], axis=0)

    # Split the data into features and target
    X_train = df_train[category_cols + continuous_cols]
    y_train = df_train[target_col]
    X_test = df_test[category_cols + continuous_cols]

    # Label encode the target column
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    sample_weights = np.array([class_weights[class_idx] for class_idx in y_train_encoded])

    # Label encode the categorical columns based on combined data
    X_combined_encoded = df_combined[category_cols + continuous_cols].copy()
    label_encoders = {}
    for col in category_cols:
        le = LabelEncoder()
        X_combined_encoded[col] = le.fit_transform(X_combined_encoded[col])
        label_encoders[col] = le

    # Split the combined encoded data back into train and test sets
    X_train_encoded = X_combined_encoded.iloc[:len(df_train), :]
    X_test_encoded = X_combined_encoded.iloc[len(df_train):, :]


    # Train the model with sample weights
    xgb_model.fit(X_train_encoded, y_train_encoded, sample_weight=sample_weights)

    # # Make predictions on the test set
    y_pred_encoded = xgb_model.predict(X_test_encoded)
    y_pred = target_encoder.inverse_transform(y_pred_encoded)

    # Calculate accuracy score
    accuracy = accuracy_score(df_test[target_col], y_pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    report = classification_report(df_test[target_col], y_pred, labels=target_encoder.classes_)
    # print(report)
    cm = confusion_matrix(df_test[target_col], y_pred)
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
    df_test['pred'] = y_pred
    return xgb_model, label_encoders, target_encoder, cm_df, report, accuracy *100, df_test
def collate(df, remove = ['Personenauto - passagier', 'Trein'], target_col = 'khvm'):
    df.khvm = df.khvm.replace({'Personenauto - passagier':'Personenauto - bestuurder'})
    for i in remove:
        df = df[df.khvm != i]
    df = df[df.khvm != 'Overig']

    df = df.replace({'Trein': 'Bus/tram/metro'})
    # df = df.replace({'Te voet': 'Fiets'})
    return df
def collatebinary(df, target_col = 'khvm'):
    df.khvm = df.khvm.replace({'Personenauto - passagier':'Personenauto - bestuurder'})
    # df = df[df.khvm != 'Personenauto - passagier']
    df = df[df.khvm != 'Overig']
    # df = df[df.khvm != 'Te voet']
    df = df.replace({'Trein': 'Bus/tram/metro'})
    # df = df.replace({'Te voet': 'Fiets'})
    df = df.replace({'Trein': 'Fiets', 'Bus/tram/metro':'Fiets', 'Te voet': 'Fiets'})

    return df
def train_xgboost(df_train, df_test, category_cols, continuous_cols, target_col):
    xgb_model = xgb.XGBClassifier(learning_rate=0.5)
    # Combine train and test data to ensure consistent encoding
    df_combined = pd.concat([df_train, df_test], axis=0)

    # Split the data into features and target
    X_train = df_train[category_cols + continuous_cols]
    y_train = df_train[target_col]
    X_test = df_test[category_cols + continuous_cols]

    # Label encode the target column
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)

    # Label encode the categorical columns based on combined data
    X_combined_encoded = df_combined[category_cols + continuous_cols].copy()

    label_encoders = {}
    for col in category_cols:
        le = LabelEncoder()
        X_combined_encoded[col] = le.fit_transform(X_combined_encoded[col])
        label_encoders[col] = le

    # Split the combined encoded data back into train and test sets
    X_train_encoded = X_combined_encoded.iloc[:len(df_train), :]
    X_test_encoded = X_combined_encoded.iloc[len(df_train):, :]

    # Train the model
    xgb_model.fit(X_train_encoded, y_train_encoded)
    # Make predictions on the test set
    y_pred_encoded = xgb_model.predict(X_test_encoded)

    # Inverse transform the encoded predictions
    y_pred = target_encoder.inverse_transform(y_pred_encoded)

    # Calculate accuracy score
    accuracy = accuracy_score(df_test[target_col], y_pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    report = classification_report(df_test[target_col], y_pred, labels = target_encoder.classes_)
    # print(report)
    cm = confusion_matrix(df_test[target_col], y_pred)
    cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)

    return xgb_model, label_encoders, target_encoder, cm_df, report, accuracy *100
def run_Felyx(F, proba = False):
    F_copy = F.copy()
    for enc in label_encoders.keys():
        F_copy[enc] = label_encoders[enc].transform(F_copy[enc])
    F_copy = F_copy[category_cols + continuous_cols]
    if proba == True:
        F[target_encoder.inverse_transform(model.classes_)] = model.predict_proba(F_copy)
    else:
        F['pred'] = target_encoder.inverse_transform(model.predict(F_copy))

    # VMA.to_pickle('SAO/VMA_Modelling/VMA_Predictions' + type)
    return F
def age_cat(df):
    age_cuts = [-1, 18, 35, 45, 65] #[-1, 15, 18, 25, 45, 65]
    df['leeftijd_cat'] = df[['leeftijd']].apply(lambda x: pd.cut(x, age_cuts, labels=[str(x) for x in age_cuts[1:]]))
    return df

pc4red = pd.read_pickle('PostcodeInfo/PCDataReduced')
# O.merge(pc4red.reset_index()[['Emb0', 'Emb1', 'Emb2', 'Postcode-4']], left_on = 'vertpc', right_on = 'Postcode-4')

target = 'khvm'
otp_cols = ['pt_duration', 'pt_distance', 'car_duration', 'car_distance', 'bike_duration', 'bike_distance', 'walk_distance'] #'distance_moved'
cost_cols = ['AVG_CHARGE', 'car_cost', 'pt_cost_GVB']#'felyx_cost', 'pt_cost'
meteo_cols = ['windspeed', 'temp', 'precip', 'precipcover']
continuous_cols = otp_cols + cost_cols + meteo_cols
category_cols = ['hour', 'CanFelyx' ,'weekdag']#,#'doel''kmotiefv', 'hhgestinkg'
category_cols += ['ovstkaart']# 'leeftijd_cat', , 'hhauto', 'opleiding']#, 'doel']#,'ovstkaart' , 'hhauto'] #geslacht', 'herkomst', 'opleiding'
# continuous_cols += ['Emb0', 'Emb1', 'Emb2']
# continuous_cols += ['aank Man',
#        'aank Huurwoning', 'aank Age0', 'aank Age1', 'aank hh0', 'aank hh1',
#        'aank build_age0', 'aank build_age1', 'aank Immigration0',
#        'aank Immigration1', 'aank size0', 'aank size1', 'vert Man',
#        'vert Huurwoning', 'vert Age0', 'vert Age1', 'vert hh0', 'vert hh1',
#        'vert build_age0', 'vert build_age1', 'vert Immigration0',
#        'vert Immigration1', 'vert size0', 'vert size1']
# category_cols += ['hhsam', 'hhwelvg', 'hhefiets', 'ovstkaart', 'hhauto', 'hhgestinkg']

O.hvm.value_counts()
O.khvm.value_counts()

jaaren = ['2018', '2021', '2020', '2019']
types = ['AADO', 'UrbanAnd'] #'AADO']#, 'UrbanAnd']
remove = []#['Trein']#, 'Personenauto - passagier' ]
modes = ['Personenauto - bestuurder', 'Fiets', 'Te voet', 'Bus/tram/metro'] #Personenauto - passagier', 'Trein']
balances = [False]
proba = False
Felyx = True
for balanced in balances:
    for type in types:
        for binary in [False]:
            # print(type, 'bal', balanced, 'bin', binary)
            O = pd.read_pickle(os.path.join('OdinData', 'OdinWrangled', 'Odin2018-2021' + type))
            # O[O.jaar.isin(jaaren)]
            # O = age_cat(O)
            # O = O.reset_index().merge(pc4red.reset_index()[['Emb0', 'Emb1', 'Emb2', 'Postcode-4']], left_on = 'vertpc', right_on = 'Postcode-4').set_index('verplid')

            F = pd.read_pickle(os.path.join('FelyxData', 'FelyxModellingData', 'felyxotpAADO'))
            F['hour'] = F.prev_time.dt.hour
            F = F.merge(pc4red.reset_index()[['Emb0', 'Emb1', 'Emb2', 'Postcode-4']], left_on = 'vertpc', right_on = 'Postcode-4')

            O = O[continuous_cols + category_cols + [target]]
            O[meteo_cols] = np.round(O[meteo_cols] / 5) * 5

            if binary == False:
                modes = [x for x in modes if x not in remove]
                O_collated = collate(O, remove = remove, target_col = target)
            else:
                modes = ['Personenauto - bestuurder', 'Fiets']
                O_collated = collatebinary(O, target)

            if type == 'AADO':
                df_train, df_test = train_test_split(O_collated, test_size = 0.25, random_state= 420)
            if type == 'UrbanAnd':

                df_train = O_collated.loc[~O_collated.index.isin(df_test.index)]

                len(set(df_test.index).difference(set(O_collated.index)))

            if balanced == True:
                model, label_encoders, target_encoder, cm_df, report, acc, xtest = train_xgboost_weighted(df_train, df_test, category_cols,continuous_cols, target)
            #
            else:
                model, label_encoders, target_encoder, cm_df, report, acc = train_xgboost(df_train, df_test, category_cols, continuous_cols, target)

            if Felyx == True:
                # print(report)

                F['leeftijd'] = 28
                F = age_cat(F)
                F['geslacht'] = 'Man'
                F['opleiding'] = 'Hoger beroepsonderwijs, universiteit'
                F['herkomst'] = 'Nederlandse achtergrond'
                F['ovstkaart'] = 'Studenten OV-chipkaart met weekendabonnement'
                F['hhauto'] = '0'

                if proba == True:
                    F_pred = run_Felyx(F, proba = True)
                    results = []
                    # for i in ['Personenauto - bestuurder']:#, 'Fiets', 'Te voet', 'Bus/tram/metro']:
                    # for i in ['Personenauto - bestuurder', 'Fiets', 'Te voet', 'Bus/tram/metro', 'Personenauto - passagier']:# 'Trein'
                    for i in modes:
                        print(i)#df_train.khvm.unique():
                        res = F_pred.iloc[:, -len(target_encoder.classes_):].sum()[i] / F_pred.iloc[:, -len(target_encoder.classes_):].sum().sum()
                        # print(round(res,2))
                        results.append(str(round(res * 100,1)))
                    print('XGBoost', '&', balanced, '&', type, '&', round(acc,2),'&',  '/'.join(results), '&' ,len(df_train), '/' , len(df_test) ,'\\\\')


                else:
                    F_pred = run_Felyx(F)
                    results = []
                    for i in modes:
                        print(i)
                        results.append(str(round(100 * (F_pred.pred.value_counts()[i] / len(F_pred)), 1)))
                    print('XGBoost', '&', balanced, '&', type, '&', round(acc,2),'&',  '/'.join(results), '&' ,len(df_train), '/' , len(df_test) , '\\\\' , '\\hline')
                    # for i in F_pred.pred.unique():
                    #     print(i, round((100 * (F_pred.pred.value_counts()[i] / len(F_pred))),2))
                print('\n')
xgb.plot_importance(model, max_num_features = 45)


O_collated.groupby(['khvm', 'weekdag']).count()
O_collated.groupby('khvm')['bike_distance'].median()

v = xtest[xtest.khvm == 'Bus/tram/metro'].groupby('pred')['pt_duration'].mean()

cor = pd.get_dummies(O_collated).corr()

F_pred.groupby('pred')['distance_moved'].mean()
F_pred.groupby('pred')['pt_duration'].mean()

O[O['khvm'] == 'Bus/tram/metro']['odin_distance'].plot(kind='hist', bins=70, alpha =0.4)
O[O['khvm'] == 'Fiets']['odin_distance'].plot(kind='hist', bins=70, alpha =0.4)

wrongbus = xtest[xtest.khvm != xtest.pred]
wrongbus = wrongbus[wrongbus['khvm'] == 'Personenauto - passagier']
wrongbus.groupby('pred').count()
wrongbus[wrongbus.pred == 'Fiets']['pt_duration'].plot(kind='hist', bins=10, alpha =0.4)
wrongbus[wrongbus.pred == 'Personenauto - bestuurder']['pt_duration'].plot(kind='hist', bins=10, alpha =0.4)
xtest[xtest.khvm == 'Personenauto - passagier']['pt_duration'].plot(kind='hist', bins=10, alpha =0.4)
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.title('Age Histogram')
# plt.show()

corF = pd.get_dummies(F_pred[category_cols+ continuous_cols + ['pred']]).corr()

walked = F_pred[F_pred.pred == 'Te voet'].dropna()[category_cols+ continuous_cols].walk_distance.quantile(0.5)
walkedO = O[O.khvm == 'Te voet'].walk_distance.quantile(0.5)
O[O.khvm == 'Fiets'].walk_distance.quantile(0.5)

F_pred.walk_distance.quantile(0.5)

print(report)

for i in cm_df.columns:
    print(sum(cm_df[i]))

for i in range(len(cm_df)):
    print(sum(cm_df.iloc[i, :]))


# if balanced == True:
#     type += 'balanced'
# F_pred.to_pickle(os.path.join('Exp1', 'Predictions','X' + type))
# cm_df
(F_pred[F_pred['Personenauto - bestuurder'] > F_pred['Fiets']]) / (len(F_pred)))

F_pred[F_pred.car_cost.isna()].plot(alpha = 0.1)

x ='khvm'
y = 'opleiding'
y = 'leeftijd_cat'

for z in O[y].unique():
    print(z)
    filtered_df = O[O[y] == z]
    class_count = filtered_df[x].value_counts()#['Bus/tram/metro']
    total_count = len(filtered_df)
    percentage = (class_count / total_count) * 100
    print(percentage)

for i in [['2018', '2019'], ['2020']]:
    a =O[O.jaar.isin(i)]
    b = a.groupby('khvm').count()
    print(b['geslacht']['Bus/tram/metro' ]/ len(a))

    print(b.iloc[:,10:13 ])
