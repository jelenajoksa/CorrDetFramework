import argparse
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as po

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



def get_data(config_path):
    config = read_params(config_path)
    data_pred_path = config['processed_data']['predicted']
    df_pred = pd.read_csv(data_pred_path, sep=',', encoding='utf-8')
    data_path = config['processed_data']['cutoff']
    df_cutoff = pd.read_csv(data_path, sep=',', encoding='utf-8')
    biggest = df_pred.nlargest(3, 'p')
    lowest = df_pred.nsmallest(3, 'p')
    df_pred['diff'] = abs(df_pred['p'] - 0.5)
    df_pred = df_pred.sort_values('diff')
    print(df_pred.loc[(df_pred['p']> 0.8) & (df_pred['p_bin'] == 1)])
    ''' 
    print(len(df_pred[df_pred['p'] <= 0.2]))
    print(len(df_pred[df_pred['p_bin'] == 0]))
    print(len(df_pred[df_pred['p_bin'] == 1]))
    '''
    # Get the first three rows (the three closest to 0.5)
    middle = df_pred.head(3)
    return df_pred, df_cutoff, biggest, lowest, middle


def plot_predictions(config_path):
    config = read_params(config_path)
    df_pred, df_all , biggest, lowest, middle = get_data(config_path)

    lowest_c = lowest['company'].tolist()
    biggest_c = biggest['company'].tolist()
    middle_c = middle['company'].tolist()

    df_normal = pd.DataFrame()

    #lowest
    for n_company in lowest_c:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal = pd.concat([df_normal, df_icompany])

    df_normal = df_normal.T
    df_normal.columns = df_normal.iloc[0, :]
    df_normal = df_normal.iloc[1:-1, :]

    fig = px.line(df_normal, x=df_normal.index, y=df_normal.columns, title='Companies with lowest predictions probabilities',width=1300, height=1000)
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
    fig.write_html('plots/cm/lowest_pred.html')



    df_normal = pd.DataFrame()

    for n_company in biggest_c:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal = pd.concat([df_normal, df_icompany])

    df_normal = df_normal.T
    df_normal.columns = df_normal.iloc[0, :]
    df_normal = df_normal.iloc[1:-1, :]

    fig = px.line(df_normal, x=df_normal.index, y=df_normal.columns, title='Companies with highest predictions probabilities', width=1300, height=1000)
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
    fig.write_html('plots/cm/biggest_pred.html')



    df_normal_t = pd.DataFrame()

    for n_company in middle_c:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal_t = pd.concat([df_normal_t, df_icompany])

    df_normal_t = df_normal_t.T
    df_normal_t.columns = df_normal_t.iloc[0, :]
    df_normal_t = df_normal_t.iloc[1:-1, :]

    #for column in df_normal.columns:
    fig = px.line(df_normal_t, x=df_normal_t.index, y=df_normal_t.columns, title='Companies with predictions probabilities closest to 0.5',width=1300, height=1000)
    fig.for_each_trace(lambda trace: trace.update(visible="legendonly")
    if trace.name in df_normal_t.columns else ())
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
    fig.write_html('plots/cm/05_pred.html')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    plot_predictions(args.config)
    get_data(args.config)
