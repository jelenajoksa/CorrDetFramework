import argparse
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import tsfresh
from tsfresh import extract_relevant_features
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import settings
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy
from sklearn.metrics import confusion_matrix

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_label_path = config['processed_data']['labeled_data']
    df_labeled = pd.read_csv(data_label_path, sep=',', encoding='utf-8')
    return df_labeled


def get_data_zcore(config_path):
    df_all = get_data(config_path)
    df_all_zcore = df_all.iloc[:,2:]
    arr_df = df_all_zcore.to_numpy()
    df_all_zcore = TimeSeriesScalerMeanVariance().fit_transform(arr_df)
    n, m, k = df_all_zcore.shape
    df_all_zcore = np.reshape(df_all_zcore, (n, m))
    df_all_zcore = pd.DataFrame(df_all_zcore)
    now = (pd.to_datetime('2011-09-15 00:00:00')).to_period('m')
    M = pd.period_range(now-104, freq='M', periods=209).strftime('%Y-%m').tolist()
    df_all_zcore.columns = M
    df_all_zcore['company'] = df_all['company']
    df_all_zcore['label'] = df_all['label']
    return df_all_zcore

def feature_extraction(config_path):
    data_label_zcore = get_data_zcore(config_path)
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
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, test_index in split.split(f, y, company):
        X_train, X_test = f.iloc[train_index], f.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        company_train, company_test = company.iloc[train_index], company.iloc[test_index]
    print("selecting features...")
    train_features_selected = select_features(X_train, y_train, fdr_level=0.0001, n_jobs=0)
    print("selected {} features.".format(len(train_features_selected.columns)))
    x_train = train_features_selected.copy()
    x_test = X_test[train_features_selected.columns].copy()
    #print(y_train.values, y_test.values)
    return x_train, y_train, x_test, y_test, company_train, company_test




def classification(config_path):
    df_all = get_data(config_path)
    x_train, y_train, x_test, y_test, company_train, company_test = feature_extraction(config_path)
    classifiers = [
        # LR( C = 10 ),
        # LR( C = 1 ),
        # LR( C = 0.1 ),

        make_pipeline(StandardScaler(), LR(C=30)),
        # make_pipeline( StandardScaler(), LR( C = 10 )),
        # make_pipeline( StandardScaler(), LR( C = 30 ))

        # LDA(),
        RF(n_estimators=100, min_samples_leaf=5)
    ]
    for clf in classifiers:
        clf.fit( x_train, y_train )
        p = clf.predict_proba( x_test )[:,1]
        p_bin = clf.predict( x_test )
        cm = confusion_matrix(y_test, p_bin)
        auc = AUC( y_test, p )
        acc = accuracy( y_test, p_bin )
        print( "AUC: {:.2%}, accuracy: {:.2%} \n\n{}\n\n".format( auc, acc, clf ))
        print('Confusion matrix for {}\n\n'.format(clf))
        print(cm)
        df_cm = pd.DataFrame()
        df_cm['p'] = p
        df_cm['p_bin'] = p_bin
        df_cm['y'] = y_test.values
        df_cm['company'] = company_test.values
        df_cm.to_csv('data/data/confusion_matrix.csv', index=False)
        cm_diff = df_cm[ df_cm['p_bin'] != df_cm['y'] ]
        cm_diff.to_csv('data/data/confusion_matrix_diff.csv', index=False)


def plot_mistakes(config_path):
    config = read_params(config_path)
    df_all = get_data(config_path)
    data_label_path = config['processed_data']['mistakes']
    df_mistakes = pd.read_csv(data_label_path, sep=',', encoding='utf-8')
    mistakes_comp = df_mistakes['company'].tolist()

    df_normal = pd.DataFrame()

    for n_company in mistakes_comp:
        df_icompany = df_all[df_all['company'] == n_company]
        df_normal = pd.concat([df_normal, df_icompany])

    df_normal = df_normal.T
    df_normal.columns = df_normal.iloc[0, :]
    df_normal = df_normal.iloc[1:, :]

    fig = px.line(df_normal, x=df_normal.index, y=df_normal.columns, title='Vzorec podjetij iz izbrane skupine')
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
    if trace.name in df_normal.columns else ())
    fig.add_vline(x='2004-12', line_dash="dash", opacity=0.5)
    fig.add_vrect(x0='2003-01', x1='2004-12', annotation_text="ROP", annotation_position="top right",
                  annotation_textangle=90, line_width=0, opacity=0.05)
    fig.add_vrect(x0='2003-01', x1='2008-11', annotation_text="JANSA", annotation_position="top right",
                  annotation_textangle=90, line_width=0, opacity=0.05)
    fig.add_vline(x='2008-11', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2012-02', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2014-09', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2013-03', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2018-09', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2020-03', line_dash="dash", opacity=0.5)
    fig.add_vrect(x0='2004-12', x1='2012-02', annotation_text="PAHOR", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0='2008-11', x1='2013-03', annotation_text="JANSA", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="blue", opacity=0.05)
    fig.add_vrect(x0='2012-02', x1='2014-09', annotation_text="BRATUSEK", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="yellow", opacity=0.05)
    fig.add_vrect(x0='2013-03', x1='2018-09', annotation_text="CERAR", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="pink", opacity=0.05)
    fig.add_vrect(x0='2014-09', x1='2020-03', annotation_text="SAREC", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="black", opacity=0.05)
    fig.add_vrect(x0='2018-09', x1='2020-05', annotation_text="JANSA", annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="orange", opacity=0.05)
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    #feature_extraction(args.config)
    #classification(args.config)
    plot_mistakes(args.config)

    '''
    AUC: 90.09%, accuracy: 79.44% 
    
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression(C=30))])
    
    
    Confusion matrix for Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression(C=30))])
    
    
    [[144  36]
     [ 38 142]]
    AUC: 89.08%, accuracy: 81.11% 
    
    RandomForestClassifier(min_samples_leaf=5)
    
    
    Confusion matrix for RandomForestClassifier(min_samples_leaf=5)
    
    
    [[143  37]
     [ 31 149]]
    '''
