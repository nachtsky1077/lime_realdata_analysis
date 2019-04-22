from lime.lime_tabular import LimeTabularExplainer
from sqf_logistic_regression import data_preprocessing, run_model
from sklearn.metrics import accuracy_score

def explain_tabular(data, explainer, black_model, labels, num_features=2, num_samples=100, verbose=1):
    '''
    :data: the data point we would like to explain
    :explainer: a lime tabular explainer
    :black_model:
    :num_features: number of features included in the explanation
    :num_samples: number of samples to generate used to train local weighted regression model
    '''
    exp = explainer.explain_instance(data, black_model.predict_proba, num_features=num_features, num_samples=num_samples, labels=labels)

    if verbose:
        print(exp.as_list()) 
    
    return exp

def do_explanation(data, X, y, iteration=10):
    X_train, y_train, X_test, y_test, lr = run_model(X, y)
    explainer = LimeTabularExplainer(training_data=X_train, 
                                     training_labels=y_train, 
                                     feature_selection='lasso_path',
                                     verbose=False, 
                                     mode='classification', 
                                     categorical_features=None, 
                                     categorical_names=None)

    y_pred = lr.predict(X_test)
    print(y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    res = []

    for iter in range(iteration):
        curr_explanation = explain_tabular(data, explainer, lr, labels=(0, 1), num_features=5, num_samples=100, verbose=0).as_list()
        print(curr_explanation)
        res.append(curr_explanation)
    return res

def analyze_lime_tabular():
    X, y = data_preprocessing()
    exp_item = (X.astype(float).loc[0, :]).values
    res = do_explanation(exp_item, X, y, 10)

    used_feature_idx = [35, 31, 2, 5, 6, 42, 14]
    for idx in used_feature_idx:
        print(X.columns[idx])

    return res

X, y = data_preprocessing()
X_train, y_train, X_test, y_test, lr = run_model(X, y)
explainer = LimeTabularExplainer(training_data=X_train, 
                                 training_labels=y_train, 
                                 feature_selection='lasso_path',
                                 verbose=False, 
                                 mode='classification', 
                                 categorical_features=None, 
                                 categorical_names=None)
y_pred = lr.predict(X_test)
print(y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
res = []
exp_item = (X.astype(float).loc[106024, :]).values
curr_explanation = explain_tabular(exp_item, explainer, lr, labels=(0, 1), num_features=5, num_samples=100, verbose=0).as_list()




if __name__=='__main__':
    analyze_lime_tabular()
