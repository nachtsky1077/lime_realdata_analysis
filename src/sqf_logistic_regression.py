import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle
from tqdm import tqdm

use_names = ['rf_vcrim', 'rf_othsw', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng',
             'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc',
             'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge',
             'cs_other', 'ac_incid', 'ac_time',  'rf_knowl', 'ac_stsnd', 'ac_other',
             'rf_furt',  'rf_bulg',  'frisked',  'searched', 'sex',      
             'race',     'age'  ]

res_names = [ 'contrabn',  'pistol',   'riflshot', 'asltweap', 'knifcuti', 'machgun',  'othrweap']


def load_data(filename='data/stop_and_frisk/2012.csv'):
    sqf_2012 = pd.read_csv(filename)
    sqf_2012 = sqf_2012[use_names + res_names]
    sqf_2012 = sqf_2012.dropna(how='any')
    return sqf_2012

def cat_onehot(df_sqf, cat_feat_name=['sex', 'race']):
    # encode category features
    cat_features = df_sqf[cat_feat_name]
    oh_enc = OneHotEncoder()
    oh_enc.fit(cat_features)
    cat_feats_array = oh_enc.transform(cat_features).toarray()
    cat_feats = pd.DataFrame(cat_feats_array, columns=oh_enc.get_feature_names())
    return cat_feats

def calc_y(label):
    '''
    known model:
    S = 3 * 1_ps + 1 * 1_as + 1 * 1_bulge
    ps: cs_objcs
    as: ac_stsnd
    bulge: cs_bulge | rf_bulge
    '''
    y = []
    for i in label.index:
        curr_label = 0
        for weap_cat in res_names:
            curr_label |= int(label.loc[i, weap_cat])
        y.append(curr_label)
    return np.array(y)

def fit_model(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


def data_preprocessing():
    try:
        X = pickle.load(open('data/stop_and_frisk/sqf_X_2012.pickle', 'rb'))
        y = pickle.load(open('data/stop_and_frisk/sqf_y_2012.pickle', 'rb'))
    except:
        feats = load_data('data/stop_and_frisk/2012.csv')
        feats.replace(to_replace=['Y'], value=1, inplace=True)
        feats.replace(to_replace=['N'], value=0, inplace=True)
        feats.replace(to_replace=['**'], value=np.nan, inplace=True)
        cat_feats = cat_onehot(feats)
        X = pd.concat([feats, cat_feats], axis=1)
        X = X.drop('sex', axis=1)
        X = X.drop('race', axis=1)
        X.dropna(inplace=True)
        label = X[res_names]
        for res_name in res_names:
            X = X.drop(res_name, axis=1)
        y = calc_y(label)

        pickle.dump(X, open('data/stop_and_frisk/sqf_X_2012.pickle', 'wb'))
        pickle.dump(y, open('data/stop_and_frisk/sqf_y_2012.pickle', 'wb'))
    return X, y

def run_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = fit_model(X_train, y_train)

    train_accuracy = lr.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, lr.predict(X_test))
    print('train accuracy:', train_accuracy)
    print('test_accuracy:', test_accuracy)

    return X_train.astype(float).values, y_train, X_test.astype(float).values, y_test, lr

X, y = data_preprocessing()








