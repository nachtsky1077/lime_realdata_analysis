import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics


feature_columns = ['sex', 'age', 'race', 'priors_count']
label_column = ['score_text']

compas_scores = pd.read_csv('data/compas/compas-scores.csv')
compas_scores = compas_scores[feature_columns + label_column].dropna(how = 'any')


X = compas_scores[feature_columns]
y = compas_scores[label_column]

# expand feature 'sex' and 'race' to categorical feature using OneHotEncoder
onehot_enc = OneHotEncoder()
onehot_enc.fit(X[['sex', 'race']])
expanded_feature = onehot_enc.transform(X[['sex', 'race']]).toarray()
feature_names = onehot_enc.get_feature_names()
race_sex_expanded = pd.DataFrame(expanded_feature, index = X.index, columns = feature_names)

# append the categorical features to the dataset
X = pd.concat([X[['age', 'priors_count']], race_sex_expanded], axis = 1)

# encode the labels
label_enc = LabelEncoder()
label_enc.fit(y)
label_enc.classes_
y_encoded = pd.DataFrame(label_enc.transform(y), index = X.index)
y_encoded.columns = ['risk_score']

# reindex
X.index = range(X.shape[0])
y_encoded.index = range(y_encoded.shape[0])

# downsampling the label = 1 class
class_1_index = np.where(y_encoded.risk_score == 1)[0]
class_1_index_subsample = np.random.choice(class_1_index, size = 3000, replace = False)
y_encoded_bal = y_encoded.drop(class_1_index_subsample, axis = 0)
X_bal = X.drop(class_1_index_subsample, axis = 0)

# split train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_encoded_bal, test_size = 0.20, shuffle  = True)


# train a RF classifier
rf = RandomForestClassifier(n_estimators = 500)
rf.fit(X_train, y_train)

# check model on test set
pred = rf.predict(X_test)
sklearn.metrics.f1_score(y_test, pred, average = None)
report = sklearn.metrics.classification_report(y_test, pred, output_dict = True)

