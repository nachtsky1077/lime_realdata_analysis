import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics
import math
import pickle
import lime.lime_tabular
import xgboost

def calc_days_in_jail(row):
    jail_in = pd.to_datetime(row['c_jail_in'])
    jail_out = pd.to_datetime(row['c_jail_out'])
    days_count = (jail_out - jail_in).days
    row['days_in_jail'] = days_count
    return row
        
def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

feature_columns = ['age', 'c_charge_degree', 'race', 
                   'sex', 'priors_count', 'days_b_screening_arrest', 
                   'juv_misd_count', 'juv_other_count', 'juv_fel_count',
                   'c_jail_in', 'c_jail_out']

                   
#label_column = ['is_recid']
label_column = ['score_text']


compas_scores = pd.read_csv('data/compas/compas-scores.csv')
compas_scores = compas_scores[feature_columns + label_column].dropna(how = 'any')
#compas_scores = compas_scores[compas_scores.is_recid != -1]
compas_scores = compas_scores[compas_scores.c_charge_degree != 'O']
#compas_scores = compas_scores[compas_scores.days_b_screening_arrest <= 30]
#compas_scores = compas_scores[compas_scores.days_b_screening_arrest >= -30]

compas_scores = compas_scores.apply(calc_days_in_jail, axis = 1)

X = compas_scores[['race', 'sex', 'c_charge_degree', 'age', 'priors_count', 'days_b_screening_arrest',
                  'juv_misd_count', 'juv_other_count', 'juv_fel_count',
                  'days_in_jail']].values
y = compas_scores[label_column].values

categorical_features_index = [0, 1, 2]
categorical_features = ['race', 'sex', 'c_charge_degree']
categorical_names = {}
for feature in categorical_features_index:
    le = LabelEncoder()
    le.fit(X[:, feature])
    X[:, feature] = le.transform(X[:, feature])
    categorical_names[feature] = le.classes_

X = X.astype(float)

# encode the labels
label_enc = LabelEncoder()
label_enc.fit(y)
label_enc.classes_
y_encoded = label_enc.transform(y)


# downsampling the label = 1 class
class_1_index = np.where(y_encoded == 1)[0]
class_1_index_subsample = np.random.choice(class_1_index, size = 2500, replace = False)
y_encoded_bal = np.delete(y_encoded, class_1_index_subsample, 0)
X_bal = np.delete(X, class_1_index_subsample, 0)

# split train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_encoded_bal, test_size = 0.20, shuffle  = True)

# save pickle dump
#pickle_dump(X_bal, 'data/compas/processed/X_balanced.pickle')
#pickle_dump(y_encoded_bal, 'data/compas/processed/y_balanced.pickle')

onehot_enc = OneHotEncoder(categorical_features = categorical_features_index)
onehot_enc.fit(X)
X_train_encoded = onehot_enc.transform(X_train)

# train a RF classifier
rf = RandomForestClassifier(n_estimators = 500, 
                            max_depth = None, 
                            min_samples_split = 2)
rf.fit(X_train_encoded, y_train)

# train a gb classifier
gb = GradientBoostingClassifier()
gb.fit(X_train_encoded, y_train)

# train a gbtree using xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(X_train_encoded, y_train)

#pickle_dump(rf, 'model/compas_rf_blackbox.pickle')

sklearn.metrics.accuracy_score(y_test, rf.predict(onehot_enc.transform(X_test)))
sklearn.metrics.accuracy_score(y_test, gb.predict(onehot_enc.transform(X_test)))
sklearn.metrics.accuracy_score(y_test, gbtree.predict(onehot_enc.transform(X_test)))

# trainig error
pred_train = rf.predict(onehot_enc.transform(X_train))
sklearn.metrics.f1_score(y_train, pred_train, average = None)
report_train = sklearn.metrics.classification_report(y_train, pred_train, output_dict = True)

# check model on test set
pred = rf.predict(onehot_enc.transform(X_test))
sklearn.metrics.f1_score(y_test, pred, average = None)
report = sklearn.metrics.classification_report(y_test, pred, output_dict = True)

# lime explanation
predict_fn = lambda x: gbtree.predict_proba(onehot_enc.transform(x)).astype(float)

feature_names = ['race', 'sex', 'c_charge_degree', 'age', 'priors_count', 'days_b_screening_arrest',
                  'juv_misd_count', 'juv_other_count', 'juv_fel_count',
                  'days_in_jail']

categorical_features_index = [0, 1, 2]
class_names = label_enc.classes_
explainer = lime.lime_tabular.LimeTabularExplainer(X_train ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features_index, 
                                                   categorical_names=categorical_names, kernel_width=3)

np.random.seed(1)
i = 200
gbtree.predict(onehot_enc.transform(X_test))[i]
iteration = 10
race_importance = []
for it in range(iteration):
    exp = explainer.explain_instance(X_test[i], predict_fn, num_features=10, labels = (0, 1, 2))
    exp_list = exp.as_list(0)
    for item in exp_list:
        if 'African' in item[0]:
            race_importance.append(item[1])
            break
print(race_importance)
