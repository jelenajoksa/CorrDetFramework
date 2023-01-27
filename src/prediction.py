import numpy as np
import pandas as pd
import argparse
import yaml
import tsfresh
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import settings
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tslearn.preprocessing import TimeSeriesScalerMeanVariance



def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config['processed_data']['cutoff']
    df_cutoff= pd.read_csv(data_path, sep=',', encoding='utf-8')
    return df_cutoff


def get_data_zcore(config_path):
    df_all = get_data(config_path)
    df_all_zcore = df_all.iloc[:,1:-1]
    arr_df = df_all_zcore.to_numpy()
    df_all_zcore = TimeSeriesScalerMeanVariance().fit_transform(arr_df)
    n, m, k = df_all_zcore.shape
    df_all_zcore = np.reshape(df_all_zcore, (n, m))
    df_all_zcore = pd.DataFrame(df_all_zcore)
    now = (pd.to_datetime('2011-09-15 00:00:00')).to_period('m')
    M = pd.period_range(now-104, freq='M', periods=209).strftime('%Y-%m').tolist()
    df_all_zcore.columns = M
    df_all_zcore['company'] = df_all['podjetje']
    #df_all_zcore['label'] = df_all['label']
    print(df_all_zcore)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    get_data_zcore(args.config)


'''
df_num_z_score = 'data/processed/df_num_z_score.csv'
df_num_z_score = pd.read_csv(df_num_z_score, sep=',')
df_num_z_score

now = (pd.to_datetime('2011-09-15 00:00:00')).to_period('m')
M = pd.period_range(now-104, freq='M', periods=209).strftime('%Y-%m').tolist()
df_num_z_score.columns = M
df_num_z_score = df_num_z_score.iloc[:,:-2]


df_num_z_score.shape


df_num_z_score = df_num_z_score.stack()
df_num_z_score.index.rename([ 'id', 'time' ], inplace = True )
df_num_z_score = df_num_z_score.reset_index()

settings = EfficientFCParameters()
f= tsfresh.extract_features( df_num_z_score , column_id = 'id', default_fc_parameters =settings, column_sort = 'time', n_jobs = 0)

impute(f)
assert f.isnull().sum().sum() == 0

col_list = list(train_features_selected.columns)
#col_list = col_list[1:]

x_test = f.loc[:,col_list]
x_test = x_test.values


for clf in classifiers:
	clf.fit( x_train, y_train )
	p = clf.predict_proba( x_test )[:,1]
	p_bin = clf.predict( x_test )
	#cm = confusion_matrix(y_test, p_bin)
	#auc = AUC( y_test, p )
	#acc = accuracy( y_test, p_bin )
	#print( "AUC: {:.2%}, accuracy: {:.2%} \n\n{}\n\n".format( auc, acc, clf ))
	#print('Confusion matrix for {}\n\n'.format(clf))
	#print(cm)
	print(p)

len(p)
len(p[p>0.8] )
len(p[p<0.5] )

'''