import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer

use_names = ['rf_vcrim', 'rf_othsw', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng',
             'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc',
             'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge',
             'cs_other', 'ac_incid', 'ac_time',  'rf_knowl', 'ac_stsnd', 'ac_other',
             'rf_furt',  'rf_bulg',  'sex',      'race']

cat_features = ['rf_vcrim', 'rf_othsw', 'rf_attir', 'cs_objcs', 'cs_descr', 'cs_casng',
                'cs_lkout', 'rf_vcact', 'cs_cloth', 'cs_drgtr', 'ac_evasv', 'ac_assoc',
                'cs_furtv', 'rf_rfcmp', 'ac_cgdir', 'rf_verbl', 'cs_vcrim', 'cs_bulge',
                'cs_other', 'ac_incid', 'ac_time',  'rf_knowl', 'ac_stsnd', 'ac_other',
                'rf_furt',  'rf_bulg',  'sex',      'race']

res_names2 = ['frisked',  'searched']
res_names = [ 'contrabn',  'pistol',   'riflshot', 'asltweap', 'knifcuti', 'machgun',  'othrweap']

#res_names = ['frisked', 'searched']

def load_data(columns, filename='data/stop_and_frisk/2012.csv'):
    sqf_2012 = pd.read_csv(filename)
    sqf_2012 = sqf_2012[columns]
    sqf_2012 = sqf_2012.dropna(how='any')
    return sqf_2012

def process_categorical_feature(df_sqf, cat_feat_name):
    cat_names = {}
    #store feature mapping
    for feature in cat_feat_name:
        le = LabelEncoder()
        le.fit(df_sqf.loc[:, feature])
        df_sqf.loc[:, feature] = le.transform(df_sqf.loc[:, feature])
        cat_names[feature] = le.classes_
    return cat_names
'''
# encode category features
cat_features = df_sqf[cat_feat_name]
oh_enc = OneHotEncoder()
oh_enc.fit(cat_features)
cat_feats_array = oh_enc.transform(cat_features).toarray()
cat_feats = pd.DataFrame(cat_feats_array, columns=oh_enc.get_feature_names())
'''

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
        for res_name in res_names:
            val = 1 if label.loc[i, res_name] == 'Y' else 0
            curr_label |= val
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
        categorical_names = pickle.load(open('data/stop_and_frisk/sqf_X_categorical_names.pickle', 'rb'))
    except:
        X = load_data(columns=use_names, filename='data/stop_and_frisk/2012.csv')
        y_raw = load_data(columns=res_names, filename='data/stop_and_frisk/2012.csv')
        X.replace(to_replace=['**'], value=np.nan, inplace=True)
        categorical_names = process_categorical_feature(X, cat_features)
        X.dropna(inplace=True)
        y = calc_y(y_raw)

        pickle.dump(X, open('data/stop_and_frisk/sqf_X_2012.pickle', 'wb'))
        pickle.dump(y, open('data/stop_and_frisk/sqf_y_2012.pickle', 'wb'))
        pickle.dump(categorical_names, open('data/stop_and_frisk/sqf_X_categorical_names.pickle', 'wb'))
    return X, y, categorical_names

def run_model(X, y, model='lr'):
    X = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    onehot_enc = OneHotEncoder(categorical_features=range(len(use_names)))
    onehot_enc.fit(X)
    encoded_X_train = onehot_enc.transform(X_train)

    if model == 'lr':
        clf = LogisticRegression()
    elif model == 'rf':
        clf = RandomForestClassifier(n_estimators=500)

    clf = fit_model(encoded_X_train, y_train)

    train_accuracy = clf.score(encoded_X_train, y_train)
    test_accuracy = accuracy_score(y_test, clf.predict(onehot_enc.transform(X_test)))
    print('train accuracy:', train_accuracy)
    print('test_accuracy:', test_accuracy)

    return clf, X_train.values, X_test.values, y_train, y_test, onehot_enc


#if __name__ == '__main__':

X, y, categorical_names = data_preprocessing()
blackbox_model, X_train, X_test, y_train, y_test, oh_enc = run_model(X, y, 'rf')
predict_fn = lambda x: blackbox_model.predict_proba(oh_enc.transform(x))
np.random.seed(1)
explainer = LimeTabularExplainer(X_train, class_names=['no_weapon', 'weapon'], feature_names=use_names,
                                 categorical_features=range(len(use_names)), categorical_names=categorical_names,
                                 mode='classification')
y_pred = predict_fn(X_test)
has_weapon_idx = []
for i in range(y_pred.shape[0]):
    if y_pred[i, 0] < y_pred[i, 1]:
        has_weapon_idx.append(i)

idx = has_weapon_idx[100]
idx = has_weapon_idx[350]
fi = dict()
for i in range(100):
    exp = explainer.explain_instance(X_test[idx], predict_fn, num_features=8, labels=[0, 1])
    #print('test sample {} prediction: {}, explanation for its predicted class:'.format(i, y_pred[idx]), 
    #       exp.as_list(y_pred[idx, 0] < y_pred[idx, 1]))
    for item in exp.as_list(1):
        if item[0] not in fi:
            fi[item[0]] = []
        fi[item[0]].append(item[1])

for feature in fi.keys():
    temp = np.array(fi[feature])
    print('*' * 20)
    print('feature name:{}'.format(feature))
    print('count:{}'.format(temp.shape[0]))
    print('std:{}'.format(temp.std()))
    print('max:{}'.format(temp.max()))
    print('min:{}'.format(temp.min()))
    print('ave:{}'.format(temp.mean()))

for idx in range(0, 100, 2):
    exp = explainer.explain_instance(X_test[idx], predict_fn, num_features=8, labels=[0, 1])
    print('test sample {} prediction: {}, explanation for its predicted class:'.format(i, y_pred[has_weapon_idx[idx]]), 
           exp.as_list(1))


