import argparse
import yaml
import numpy as np
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as auc, accuracy_score as accuracy, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

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
    return df_labeled, df_cutoff


def get_data_zcore(config_path):
    df_labeled, df_cutoff = get_data(config_path)
    df_labeled_zcore = df_labeled.iloc[:,2:]
    arr_df = df_labeled_zcore.to_numpy()
    df_labeled_zcore = TimeSeriesScalerMeanVariance().fit_transform(arr_df)
    n, m, k = df_labeled_zcore.shape
    df_labeled_zcore = np.reshape(df_labeled_zcore, (n, m))
    df_labeled_zcore = pd.DataFrame(df_labeled_zcore)
    now = (pd.to_datetime('2011-09-15 00:00:00')).to_period('m')
    M = pd.period_range(now-104, freq='M', periods=209).strftime('%Y-%m').tolist()
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

def feature_extraction(config_path):
    data_label_zcore, df_cutoff_zscore = get_data_zcore(config_path)
    y = data_label_zcore.label
    company = data_label_zcore.company
    data_zcore = data_label_zcore.iloc[:,:-2]
    data_zcore = data_zcore.stack()
    data_zcore.index.rename([ 'id', 'time' ], inplace = True )
    data_zcore = data_zcore.reset_index()
    settings = EfficientFCParameters()
    f= tsfresh.extract_features( data_zcore , column_id = 'id', default_fc_parameters =settings, column_sort = 'time', n_jobs = 0)
    impute(f)
    assert f.isnull().sum().sum() == 0
    split = StratifiedKFold(n_splits=10)
    for train_index, test_index in split.split(f, y, company):
        X_train, X_test = f.iloc[train_index], f.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        company_train, company_test = company.iloc[train_index], company.iloc[test_index]
    print("selecting features...")
    train_features_selected = select_features(X_train, y_train, fdr_level=0.0001, n_jobs=0)
    print("selected {} features.".format(len(train_features_selected.columns)))
    x_train = X_train[train_features_selected.columns].copy()
    x_test = X_test[train_features_selected.columns].copy()
    return x_train, y_train, x_test, y_test, company_train, company_test


def feature_extraction_pred(config_path):
    x_train, y_train, x_test, y_test, company_train, company_test = feature_extraction(config_path)
    data_label_zcore, df_cutoff_zscore = get_data_zcore(config_path)
    #training set for prediction:
    x_train_union = pd.concat([x_train, x_test])
    y_train_union = pd.concat([y_train, y_test])
    #dataset to be predicted:
    company_pred = df_cutoff_zscore.company
    print(company_pred)
    df_zscore = df_cutoff_zscore.iloc[:, :-1]
    df_zscore = df_zscore.stack()
    df_zscore.index.rename(['id', 'time'], inplace=True)
    df_zscore = df_zscore.reset_index()
    print(df_zscore)
    settings = EfficientFCParameters()
    f1 = tsfresh.extract_features( df_zscore , column_id = 'id', default_fc_parameters = settings, column_sort = 'time', n_jobs = 0)
    impute(f1)
    assert f1.isnull().sum().sum() == 0
    x_pred = f1[x_train.columns].copy()
    return x_train_union, y_train_union, x_pred, company_pred


def classification(config_path):
    x_train, y_train, x_test, y_test, company_train, company_test = feature_extraction(config_path)

    classifiers = [
        make_pipeline(StandardScaler(), LR(C=30)),
        RF(n_estimators=100, min_samples_leaf=5)
    ]

    for clf in classifiers:
        clf.fit( x_train, y_train )
        p = clf.predict_proba( x_test )[:,1]
        p_bin = clf.predict( x_test )
        cm = confusion_matrix(y_test, p_bin)
        roc_auc = auc( y_test, p_bin )
        acc = accuracy( y_test, p_bin )
        prec = precision_score(y_test, p_bin)
        rec = recall_score(y_test, p_bin)
        f1 = f1_score(y_test, p_bin)

        print( "AUC: {:.2%}, accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}, F1 score: {:.2%} \n\n{}\n\n".format( roc_auc, acc, prec, rec, f1, clf ))
        print('Confusion matrix for {}\n\n'.format(clf))
        print(cm)
        print("Classification Report: ",classification_report(y_test, p_bin))

        #printing out the predictions
        df_cm = pd.DataFrame()
        df_cm['p'] = p
        df_cm['p_bin'] = p_bin
        df_cm['y'] = y_test.values
        df_cm['company'] = company_test.values
        #df_cm.to_csv('data/data/predictions.csv', index=False)
        df_cm.to_csv('results/predictions.csv', index=False)

        cm_false_p = df_cm.loc[(df_cm['p_bin'] == 1) & (df_cm['y'] == 0)]
        cm_false_p.to_csv('data/data/false_pos_predictions.csv', index=False)
        cm_false_n = df_cm.loc[(df_cm['p_bin'] == 0) & (df_cm['y'] == 1)]
        cm_false_n.to_csv('data/data/false_neg_predictions.csv', index=False)
        cm_true = df_cm[df_cm['p_bin'] == df_cm['y']]
        cm_true.to_csv('data/data/true_predictions.csv', index=False)


def prediction(config_path):
    x_train_union, y_train_union, x_pred, company_pred = feature_extraction_pred(config_path)
    classifiers = [
        make_pipeline(StandardScaler(), LR(C=30)),
        RF(n_estimators=100, min_samples_leaf=5)
    ]

    for clf1 in classifiers:
        clf1.fit(x_train_union, y_train_union)
        p1 = clf1.predict_proba(x_pred)[:, 1]
        p1_bin = clf1.predict(x_pred)
        df_pred = pd.DataFrame()
        df_pred['p'] = p1
        df_pred['p_bin'] = p1_bin
        df_pred['company'] = company_pred.values
        df_pred.to_csv('data/data/all_predictions.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    get_data_zcore(args.config)
    #feature_extraction(args.config)
    classification(args.config)

    prediction(args.config)
