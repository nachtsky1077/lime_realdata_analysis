import coloredlogs
import copy
import logging
from datetime import datetime
import pprint

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics

__all__ = ['Reader', 'Trainer']


class Reader(object):

    def __init__(self, data_path):

        self._data_path = data_path
        self._compas_scores = pd.read_csv(data_path)

        logging.basicConfig(level=logging.DEBUG, format='[%(name)s]:\t%(levelname)s\t\t%(message)s')
        self._logger = logging.getLogger(__name__)
        self._default_invalid_features = ['id', 'is_recid', 'type_of_assessment', 'v_score_text',
                                  'vr_charge_degree', 'vr_charge_desc', 'r_case_number',
                                  'r_offense_date', 'v_screening_date', 'v_type_of_assessment',
                                  'r_jail_out', 'decile_score.1', 'vr_case_number',
                                  'compas_screening_date', 'r_days_from_arrest', 'r_jail_in',
                                  'last', 'num_vr_cases', 'r_charge_desc', 'c_charge_desc',
                                  'name', 'first', 'vr_offense_date', 'c_offense_date',
                                  'decile_score', 'c_arrest_date', 'num_r_cases', 'c_case_number',
                                  'age_cat']

        self._invalid_features = None
        self._valid_features = None
        self._special_features = ['sex', 'age', 'race', 'priors_count']
        self._label_name = ['score_text']

    def _get_valid_features_given_invalid(self, invalid_features):
        all_features = self._compas_scores.columns
        valid_features = list(
            set(all_features) - set(invalid_features)
        )
        return valid_features

    # just do datetime transform
    # feature / label encoding is none of Reader's business
    def preprocess_from_raw(self, invalid_features=None):

        if invalid_features is None:
            self._logger.warning('You do not specify the invalid features, '
                                 'default invalid features are as follows {}'
                                 .format(self._default_invalid_features))
            self._invalid_features = copy.deepcopy(self._default_invalid_features)
        elif not isinstance(invalid_features, list):
            self._logger.error('Invalid features should be `List` rather than {}'
                               .format(type(invalid_features)))
        else:
            self._invalid_features = invalid_features

        self._valid_features = self._get_valid_features_given_invalid(self._invalid_features)
        basic_compas_dataframe = self._compas_scores[self._valid_features]
        self._logger.info('num of records before drop na: {}'.format(len(basic_compas_dataframe)))
        basic_compas_dataframe = basic_compas_dataframe.dropna(how='any')
        self._logger.info('final valid count is : {}'.format(len(basic_compas_dataframe)))

        # basic time transformation
        self._logger.info('begin datetime processing')

        basic_datetime = datetime(2012, 12, 31)
        for i in basic_compas_dataframe.index:
            # 4time characteristics
            try:
                screening_date = basic_compas_dataframe.loc[i, ['screening_date']][0]
                c_jail_in = basic_compas_dataframe.loc[i, ['c_jail_in']][0]
                dob = basic_compas_dataframe.loc[i, ['dob']][0]
                c_jail_out = basic_compas_dataframe.loc[i, ['c_jail_out']][0]

                basic_compas_dataframe.loc[i, ['screening_date']] = \
                    (datetime.strptime(screening_date, "%Y-%m-%d") - basic_datetime).total_seconds()
                basic_compas_dataframe.loc[i, ['c_jail_in']] = \
                    (datetime.strptime(c_jail_in, "%Y-%m-%d %H:%M:%S") - basic_datetime).total_seconds()
                basic_compas_dataframe.loc[i, ['dob']] = \
                    (datetime.strptime(dob, "%Y-%m-%d") - basic_datetime).total_seconds()
                basic_compas_dataframe.loc[i, ['c_jail_out']] = \
                    (datetime.strptime(c_jail_out, "%Y-%m-%d %H:%M:%S") - basic_datetime).total_seconds()
            except TypeError as te:
                print(te)

        self._logger.info('done datetime processing')

        self._basic_compas_dataframe = basic_compas_dataframe

        self._logger.info('begin encoding features')
        # encode categorical features
        X = basic_compas_dataframe[list(set(self._valid_features) - {'score_text'})]
        self._X = pd.get_dummies(X)
        y = basic_compas_dataframe[['score_text']]

        self._logger.info('begin encoding labels')
        # encode labels
        self._y = self._get_encoded_label(y)

        return self._X, self._y

    def write_to_csv(self, dataframe=None, file_path=None):
        if file_path is None:
            self._logger.info('not specify out path, out_df.csv will be used')
            file_path = './out_df.csv'
        self._logger.info('begin writing')
        if dataframe is None:
            # write basic dataframe to file
            self._basic_compas_dataframe.to_csv(file_path)
        else:
            dataframe.to_csv(file_path)
        self._logger.info('done writing')

    def get_dataframe(self):
        return self._compas_scores


class Trainer(object):
    """
    Currently a training pipeline for RandomForest or Gradient boosting tree
    Attention:
        * accept dataframes only have numerical data or categorical data(without datetime)
        * currently assumes that the label is a categorical value
    """

    def __init__(self, dataframe, label, features, method='rf', random_seed=233, **params):
        assert method.lower() in ['rf', 'gbt'], 'only randomforest and gradient' \
                                                ' boosting tree supported up to new'

        logging.basicConfig(level=logging.INFO, format='%(name)s:\t%(levelname)s\t\t%(message)s')
        self._logger = logging.getLogger(__name__)
        # coloredlogs.install(level='DEBUG')

        self._random_seed = random_seed
        self._method = method
        self._original_df = dataframe.dropna(how='any')
        self.label_ = label
        self.features_ = features
        self._logger.info('Dataframe statisticals: ')
        self._print_basic_statistical_of_df()
        self._params = params
        self._print_method_settings()

    def _print_method_settings(self):
        self._logger.info('method using: {}'.format(self._method))
        self._logger.info('params: ')
        # pprint.pprint(self._params)
        for k, v in self._params.items():
            self._logger.info('{} = {}'.format(k, v))

    def _print_basic_statistical_of_df(self):

        self._logger.info(
            'length of the dataframe: {}'
            .format(len(self._original_df))
        )
        self._logger.info(
            'features: {}'
            .format(self.features_)
        )
        self._logger.info(
            'label: {}'.format(self.label_)
        )

    def _get_encoded_label(self, y):
        # encode labels
        label_enc = LabelEncoder()
        # np.array(y).ravel() to remove warnings
        label_enc.fit(np.array(y).ravel())
        self._logger.info('label classes: {}'.format(label_enc.classes_))

        y_encoded = label_enc.transform(np.array(y).ravel())
        y_encoded = pd.DataFrame(y_encoded, index=y.index, columns=['risk_score'])
        self._logger.info('class mapping: ')

        encode_labels = range(len(label_enc.classes_))
        labels = label_enc.inverse_transform(encode_labels)
        class_mapping = dict()
        for i in range(len(labels)):
            class_mapping[encode_labels[i]] = labels[i]
        _ = [self._logger.info(str(encode_labels[i]) + ": " + (labels[i])) for i in range(len(labels))]

        self._label_class_mapping = class_mapping
        return y_encoded

    def train_pipeline(self):
        # process categorical feature
        X = self._original_df[self.features_]
        X = pd.get_dummies(X)

        # process categorical label
        y = self._original_df[self.label_]
        y_encoded = self._get_encoded_label(y)

        # self._logger.info('X head')
        # self._logger.info(X.head())
        # self._logger.info('y head')
        # self._logger.info(y.head())
        # print(X.head())
        # print(y.head())

        # train, test split
        random_state_seed = 233
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, shuffle=True,
                                                            random_state=self._random_seed)

        if self._method == 'rf':
            rf = RandomForestClassifier(**self._params)
            rf.fit(X_train, np.array(y_train).ravel())
            self._logger.info('output results:')
            print('\nmean accuracy: {}\n'.format(round(rf.score(X_test, y_test), 3)))
            feature_importance_list = list(zip(X.columns, rf.feature_importances_))
            feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
            print('feature importances: ')
            _ = [print(importance) for importance in feature_importance_list]

            print('\noob score: ')
            print(rf.oob_score_)
            pred = rf.predict(X_test)
            f1_result = sklearn.metrics.f1_score(y_test, pred, average=None)
            print('\nf1 score results')
            for i in range(len(f1_result)):
                print('{} : {}'.format(self._label_class_mapping[i], round(f1_result[i], 3)))
            print('\nthorough output')
            report = sklearn.metrics.classification_report(y_test, pred, output_dict=True)
            for key, value in self._label_class_mapping.items():
                report[value] = report[str(key)]
                report.pop(str(key))
            pprint.pprint(report)

        elif self._method == 'gbt':
            gbt = GradientBoostingClassifier(**self._params)
            gbt.fit(X_train, np.array(y_train).ravel())
            print('\nmean accuracy: {}\n'.format(round(gbt.score(X_test, y_test), 3)))
            feature_importance_list = list(zip(X.columns, gbt.feature_importances_))
            feature_importance_list = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
            print('\nfeature importances: ')
            _ = [print(importance) for importance in feature_importance_list]
            # print('\noob score: ')
            # print(gbt.oob_score_)
            pred = gbt.predict(X_test)
            f1_result = sklearn.metrics.f1_score(y_test, pred, average=None)
            print('\nf1 score results')
            for i in range(len(f1_result)):
                print('{} : {}'.format(self._label_class_mapping[i], round(f1_result[i], 3)))
            print('\nthorough output')
            report = sklearn.metrics.classification_report(y_test, pred, output_dict=True)
            for key, value in self._label_class_mapping.items():
                report[value] = report[str(key)]
                report.pop(str(key))
            pprint.pprint(report)


if __name__ == '__main__':

    # reader = Reader('/Users/yee/Desktop/lime/lime_readdata_analysis/data/compas/compas-scores.csv')
    reader = Reader('./out_df.csv')
    basic_data_frame = reader.get_dataframe()
    basic_features = ['age', 'sex', 'race', 'priors_count',
                       'screening_date', 'c_jail_in', 'juv_other_count',
                       'dob', 'c_days_from_compas', 'juv_misd_count',
                       'juv_fel_count', 'c_jail_out', 'v_decile_score',
                       'days_b_screening_arrest', 'is_violent_recid',
                       'r_charge_degree', 'c_charge_degree']
    special_features = ['age', 'sex', 'race', 'priors_count']
    add_features = ['v_decile_score', 'r_charge_degree', 'c_charge_degree', 'is_violent_recid']
    features_selected = special_features + add_features

    trainer = Trainer(basic_data_frame,
                      ['score_text'],
                      features_selected,
                      method='gbt',
                      n_estimators=100,
                      learning_rate=0.2,
                      random_state=233)

    trainer.train_pipeline()
    # basic_data_frame = reader.preprocess_from_raw()


    # reader.write_to_csv()
    # trainer = Trainer(basic_data_frame, )
