import argparse
import yaml
import numpy as np
from numpy import *
import pandas as pd
import plotly.express as px
import plotly.offline as po
import tsfresh
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import settings
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as auc, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import copy
import seaborn as sns
import matplotlib.pyplot as plt


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)
    data_label_path = config['processed_data']['labeled_data']
    df_labeled = pd.read_csv(data_label_path, sep=',', encoding='utf-8')
    data_path = config['processed_data']['cutoff']
    df_cutoff = pd.read_csv(data_path, sep=',', encoding='utf-8')

    df_labeled_zcore = df_labeled.iloc[:, 2:]
    arr_df = df_labeled_zcore.to_numpy()
    df_labeled_zcore = TimeSeriesScalerMeanVariance().fit_transform(arr_df)
    n, m, k = df_labeled_zcore.shape
    df_labeled_zcore = np.reshape(df_labeled_zcore, (n, m))
    df_labeled_zcore = pd.DataFrame(df_labeled_zcore)
    now = (pd.to_datetime('2011-09-15 00:00:00')).to_period('m')
    M = pd.period_range(now - 104, freq='M', periods=209).strftime('%Y-%m').tolist()
    df_labeled_zcore.columns = M
    df_labeled_zcore['company'] = df_labeled['company']
    df_labeled_zcore['label'] = df_labeled['label']

    df_cutoff_zscore = df_cutoff.iloc[:, 1:-1]
    arr_df2 = df_cutoff_zscore.to_numpy()
    df_cutoff_zscore = TimeSeriesScalerMeanVariance().fit_transform(arr_df2)
    n2, m2, k2 = df_cutoff_zscore.shape
    df_cutoff_zscore = np.reshape(df_cutoff_zscore, (n2, m2))
    df_cutoff_zscore = pd.DataFrame(df_cutoff_zscore)
    df_cutoff_zscore.columns = M
    df_cutoff_zscore['company'] = df_cutoff['podjetje']
    return df_labeled_zcore, df_cutoff_zscore


def feature_selection(config_path):
    df_labeled_zcore, df_cutoff_zscore = get_data(config_path)
    y = df_labeled_zcore.label
    company = df_labeled_zcore.company
    data_zcore = df_labeled_zcore.iloc[:, :-2]
    data_zcore = data_zcore.stack()
    data_zcore.index.rename(['id', 'time'], inplace=True)
    data_zcore = data_zcore.reset_index()
    settings = EfficientFCParameters()

    f = tsfresh.extract_features(data_zcore, column_id='id', default_fc_parameters=settings, column_sort='time',
                                 n_jobs=0)
    impute(f)
    assert f.isnull().sum().sum() == 0
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in split.split(f, y, company):
        X_train, X_test = f.iloc[train_index], f.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print("selecting features...")
    train_features_selected = select_features(X_train, y_train, fdr_level=0.0001, n_jobs=0)
    print("selected {} features.".format(len(train_features_selected.columns)))
    x = f[train_features_selected.columns].copy()

    feature_list = (x.columns).to_list()
    with open('results/features.txt', 'w') as file:
        for feature in feature_list:
            file.write(feature)
            file.write('\n')

    company_pred = df_cutoff_zscore.company

    df_zscore = df_cutoff_zscore.iloc[:, :-1]
    df_zscore = df_zscore.stack()
    df_zscore.index.rename(['id', 'time'], inplace=True)
    df_zscore = df_zscore.reset_index()

    settings = EfficientFCParameters()

    f1 = tsfresh.extract_features(df_zscore, column_id='id', default_fc_parameters=settings, column_sort='time',
                                  n_jobs=0)
    impute(f1)
    assert f1.isnull().sum().sum() == 0
    x_pred = f1[x.columns].copy()
    return x, y, company, x_pred, company_pred


def model(x, y, company, x_pred, company_pred):
    random.seed(1234)
    skf = StratifiedKFold(n_splits=10, random_state=random.seed(1234))
    lst_accu_stratified = []
    clf = RF(n_estimators=100, min_samples_leaf=5, random_state=random.seed(1234))

    accuracy = cross_val_score(clf, x, y, cv=skf, scoring='accuracy', verbose=10)
    precision_score = cross_val_score(clf, x, y, cv=skf, scoring='precision', verbose=10)
    f1_results = cross_val_score(clf, x, y, cv=skf, scoring='f1', verbose=10)
    roc_auc_results = cross_val_score(clf, x, y, cv=skf, scoring='roc_auc', verbose=10)
    racall_results = cross_val_score(clf, x, y, cv=skf, scoring='recall', verbose=10)

    all_results = ['Overall Accuracy: mean {:.2%}, standard deviation {:.2%}'.format(accuracy.mean(), accuracy.std()),
                   'Overall Precision: mean {:.2%}, standard deviation {:.2%}'.format(precision_score.mean(),
                                                                                      precision_score.std()),
                   'Overall F1-Score: mean {:.2%}, standard deviation {:.2%}'.format(f1_results.mean(),
                                                                                     f1_results.std()),
                   'Overall ROC AUC: mean {:.2%}, standard deviation {:.2%}'.format(roc_auc_results.mean(),
                                                                                    roc_auc_results.std()),
                   'Overall Recall: mean {:.2%}, standard deviation {:.2%}'.format(racall_results.mean(),
                                                                                   racall_results.std())]

    with open('results/results.txt', 'w') as file:
        for result in all_results:
            file.write(result)
            file.write('\n')
    return clf, skf


def cross_val_predict(x, y, company, x_pred, company_pred):
    x = np.array(x)
    y = np.array(y)
    company = np.array(company)
    model_ = copy.deepcopy(clf)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    actual_companies = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in skf.split(x, y, company):

        train_X, train_y, train_c, test_X, test_y, test_c = x[train_ndx], y[train_ndx], company[train_ndx], x[test_ndx], \
                                                            y[test_ndx], company[test_ndx]

        actual_companies = np.append(actual_companies, test_c)
        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))
        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, actual_companies, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    plt.figure(figsize=(12.8, 6))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')


def plot_cm():
    plot_confusion_matrix(actual_classes, predicted_classes, [0, 1])


def evaluation():
    df_proba = pd.DataFrame(predicted_proba)
    df_cm = pd.DataFrame()

    df_cm['p_bin'] = predicted_classes
    df_cm['y'] = actual_classes
    df_cm['company'] = actual_companies
    df_cm['prob_0'] = df_proba.iloc[:, 0]
    df_cm['prob_1'] = df_proba.iloc[:, 1]
    # df_cm = (df_cm.loc[(df_cm['company']!=0) & (df_cm['company']!=1)])
    df_cm.to_csv('data/data/df_cm.csv', index=False)
    df_cm.to_csv('results/predictions.csv', index=False)

    cm_false_p = df_cm.loc[(df_cm['p_bin'] == 1) & (df_cm['y'] == 0)]
    cm_false_p.to_csv('data/data/false_pos_predictions.csv', index=False)
    cm_false_n = df_cm.loc[(df_cm['p_bin'] == 0) & (df_cm['y'] == 1)]
    cm_false_n.to_csv('data/data/false_neg_predictions.csv', index=False)
    cm_true = df_cm[df_cm['p_bin'] == df_cm['y']]
    cm_true.to_csv('data/data/true_predictions.csv', index=False)


# predictions on unseen data
def prediction_unseen_data():
    random.seed(1234)
    model2 = copy.deepcopy(clf)
    model2.fit(x, y)
    p_pred_0 = model2.predict_proba(x_pred)[:, 0]
    p_pred_1 = model2.predict_proba(x_pred)[:, 1]
    p_pred_bin = model2.predict(x_pred)

    df_pred = pd.DataFrame()

    df_pred['p_0'] = p_pred_0
    df_pred['p_1'] = p_pred_1
    df_pred['p_bin'] = p_pred_bin
    df_pred['company'] = company_pred.values
    df_pred.to_csv('data/data/all_predictions.csv', index=False)

    # plot fig5 in the Article:
    d = {'prob': [0.8, 0.85, 0.9, 0.95], 'no_of_examples': [2452, 1758, 975, 256]}
    dd = pd.DataFrame(data=d)

    plt.figure(figsize=(6, 4))
    plt.plot(dd['prob'], dd['no_of_examples'], '-p')
    plt.xlabel('Probability of being classified as an interesting example', fontsize=22)
    plt.ylabel('Number of examples', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rcParams.update({'font.size': 25})
    for a, b in zip(dd['prob'], dd['no_of_examples']):
        plt.text(a, b, str(b))
    plt.savefig('plots/cm/fig5.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
   
    x, y, company, x_pred, company_pred = feature_selection(args.config)
    clf, skf = model(x, y, company, x_pred, company_pred)
    actual_classes, actual_companies, predicted_classes, predicted_proba = cross_val_predict(x, y, company, x_pred, company_pred)
    plot_cm()
    evaluation()
    prediction_unseen_data()

