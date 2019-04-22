import pickle
import lime.lime_tabular
import sklearn.metrics


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(filename):
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    return df

model = load_model('model/compas_rf_blackbox.pickle')
X = load_data('data/compas/processed/X_balanced.pickle')
y = load_data('data/compas/processed/y_balanced.pickle')

