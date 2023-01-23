import pandas as pd
import argparse
import yaml
import plotly.express as px
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config['processed_data']['clustered_data']
    clustered_data = pd.read_csv(data_path, sep=',', encoding='utf-8')
    clustered_data = clustered_data.rename(columns={'podjetje': 'company'})
    return clustered_data

def get_interesting_examples(config_path):
    config = read_params(config_path)
    df_all = get_data(config_path)
    cluster_n_comp = []
    cluster_n = df_all[df_all['cluster'] == config['clustering']['cluster_example']]
    cluster_n = cluster_n.sample(20)
    cluster_n_comp = cluster_n['company'].tolist()
    df_normal = pd.DataFrame()

    for n_company in cluster_n_comp:
        df_icompany = df_all[df_all['company']==n_company]
        df_normal =  pd.concat([df_normal,df_icompany])

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

def get_data_zcore(config_path):
    config = read_params(config_path)
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
    df_all_zcore['cluster'] = df_all['cluster']
    return df_all_zcore



def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

#input je clustersd data - kmeans.csv
def similar_companies(config_path):
    config = read_params(config_path)
    df_all = get_data(config_path)
    df_all_zcore = get_data_zcore(config_path)
    #the line below is only run for the first time, then it goes under the comment
    #companies_export = pd.DataFrame()
    #two lines below are not under the comment if the line above is and vice versa
    data_path = config['processed_data']['labeled_data']
    companies_export = pd.read_csv(data_path, sep=',', encoding='utf-8')

    target_company = df_all_zcore[df_all_zcore['company'] == config['clustering']['company_example']]
    target_company = target_company.iloc[:, :-2]
    target_company = target_company.values
    target_company = target_company.T

    i, j = target_company.shape
    target_company = np.reshape(target_company, (i))

    df_num_10 = df_all_zcore.drop('cluster', axis=1)
    df_num_10 = df_num_10.drop('company', axis=1)
    df_num_10 = np.array(df_num_10)

    ed_list = []

    for ts in df_num_10:
         ed = euclidean_distance(target_company, ts)
         ed_list.append(ed)

    len(ed_list)

    df_all['corr'] = ed_list

    companies_15 = df_all.nsmallest(15, 'corr')
    companies_export = pd.concat([companies_export, companies_15])
    companies_export.to_csv('data/data/labeled_data.csv', index=False)


    #comment out this part to preview all 15 similar companies with the target one
    ''' 
    similar_15 = df_all.nsmallest(15, 'corr')
    similar_15 = similar_15.T
    similar_15.columns = similar_15.iloc[0, :]
    similar_15 = similar_15.iloc[1:, :]
    
    fig = px.line(similar_15, x=similar_15.index, y=similar_15.columns, title='Podobna podjetja')
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
    if trace.name in similar_15.columns else ())
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
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    get_data(args.config)
    #get_interesting_examples(args.config)
    similar_companies(args.config)
