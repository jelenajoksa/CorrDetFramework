import argparse
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)

    data_path = config['processed_data']['cutoff']
    df_cutoff = pd.read_csv(data_path, sep=',', encoding='utf-8')

    false_neg_path = config['processed_data']['false_neg_pred']
    df_false_neg = pd.read_csv(false_neg_path, sep=',', encoding='utf-8')

    false_pos_path = config['processed_data']['false_pos_pred']
    df_false_pos = pd.read_csv(false_pos_path, sep=',', encoding='utf-8')

    data_true_path = config['processed_data']['true_pred']
    df_true = pd.read_csv(data_true_path, sep=',', encoding='utf-8')

    data_pred_path = config['processed_data']['predicted']
    df_pred = pd.read_csv(data_pred_path, sep=',', encoding='utf-8')
    biggest = df_pred.nlargest(3, 'p_1')

    return df_cutoff, df_false_neg, df_false_pos, df_true, df_pred, biggest


def plot_cm_examples(config_path):
    df_cutoff, df_false_neg, df_false_pos, df_true, df_pred, biggest = get_data(config_path)
    print(df_true)
    false_neg_comp = df_false_neg['company'].tolist()
    false_pos_comp = df_false_pos['company'].tolist()
    true_comp = df_true['company'].tolist()

    df_normal = pd.DataFrame()

    # false_positives
    for n_company in false_pos_comp:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal = pd.concat([df_normal, df_icompany])

    df_normal = df_normal.T
    df_normal.columns = df_normal.iloc[0, :]
    df_normal = df_normal.iloc[1:-1, :]
    df_normal = df_normal.sample(n=3, axis='columns')

    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 0], mode='lines', line_width=4), row=1, col=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 1], mode='lines', line_width=4), row=2, col=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 2], mode='lines', line_width=4), row=3, col=1)
    fig.add_vline(x='2004-12', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2008-11', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2012-02', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2014-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2013-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2018-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2020-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vrect(x0='2003-01', x1='2004-12', annotation_text="ROP", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2003-01', x1='2008-11', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2004-12', x1='2012-02', annotation_text="PAHOR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0='2008-11', x1='2013-03', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="blue", opacity=0.05)
    fig.add_vrect(x0='2012-02', x1='2014-09', annotation_text="BRATUSEK", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="yellow", opacity=0.05)
    fig.add_vrect(x0='2013-03', x1='2018-09', annotation_text="CERAR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="pink", opacity=0.05)
    fig.add_vrect(x0='2014-09', x1='2020-03', annotation_text="SAREC", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="black", opacity=0.05)
    # fig.add_vrect(x0='2018-09', x1='2020-05', annotation_text="JANSA", annotation_position="top right", annotation_font_size=30,
    # annotation_textangle=90, line_width=3, fillcolor="orange", opacity=0.05)
    fig.update_layout(yaxis=dict(tickfont=dict(size=30)), yaxis2=dict(tickfont=dict(size=30)),
                      yaxis3=dict(tickfont=dict(size=30)), xaxis1=dict(tickfont=dict(size=30)),
                      xaxis2=dict(tickfont=dict(size=30)), xaxis3=dict(tickfont=dict(size=30)))
    fig.update_layout(height=900, width=2000)
    fig.show()
    fig.write_html('plots/cm/false_pos.html')

    df_normal_n = pd.DataFrame()

    # false negatives
    for n_company in false_neg_comp:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal_n = pd.concat([df_normal_n, df_icompany])

    df_normal_n = df_normal_n.T
    df_normal_n.columns = df_normal_n.iloc[0, :]
    df_normal_n = df_normal_n.iloc[1:-1, :]
    df_normal_n = df_normal_n.sample(n=3, axis='columns')

    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(go.Scatter(x=df_normal_n.index, y=df_normal_n.iloc[:, 0], mode='lines', line_width=4), row=1,
                     col=1)
    fig.append_trace(go.Scatter(x=df_normal_n.index, y=df_normal_n.iloc[:, 1], mode='lines', line_width=4), row=2,
                     col=1)
    fig.append_trace(go.Scatter(x=df_normal_n.index, y=df_normal_n.iloc[:, 2], mode='lines', line_width=4), row=3,
                     col=1)
    fig.add_vline(x='2004-12', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2008-11', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2012-02', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2014-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2013-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2018-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2020-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vrect(x0='2003-01', x1='2004-12', annotation_text="ROP", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2003-01', x1='2008-11', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2004-12', x1='2012-02', annotation_text="PAHOR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0='2008-11', x1='2013-03', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="blue", opacity=0.05)
    fig.add_vrect(x0='2012-02', x1='2014-09', annotation_text="BRATUSEK", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="yellow", opacity=0.05)
    fig.add_vrect(x0='2013-03', x1='2018-09', annotation_text="CERAR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="pink", opacity=0.05)
    fig.add_vrect(x0='2014-09', x1='2020-03', annotation_text="SAREC", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="black", opacity=0.05)
    # fig.add_vrect(x0='2018-09', x1='2020-05', annotation_text="JANSA", annotation_position="top right", annotation_font_size=30,
    # annotation_textangle=90, line_width=3, fillcolor="orange", opacity=0.05)
    fig.update_layout(yaxis=dict(tickfont=dict(size=30)), yaxis2=dict(tickfont=dict(size=30)),
                      yaxis3=dict(tickfont=dict(size=30)), xaxis1=dict(tickfont=dict(size=30)),
                      xaxis2=dict(tickfont=dict(size=30)), xaxis3=dict(tickfont=dict(size=30)))
    fig.update_layout(height=900, width=2000)
    fig.show()
    fig.write_html('plots/cm/false_neg.html')

    # true predictions
    df_normal_t = pd.DataFrame()

    for n_company in true_comp:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal_t = pd.concat([df_normal_t, df_icompany])

    df_normal_t = df_normal_t.T
    df_normal_t.columns = df_normal_t.iloc[0, :]
    df_normal_t = df_normal_t.iloc[1:-1, :]
    df_normal_t = df_normal_t.sample(n=3, axis='columns')
    # for column in df_normal.columns:
    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(go.Scatter(x=df_normal_t.index, y=df_normal_t.iloc[:, 0], mode='lines', line_width=4), row=1,
                     col=1)
    fig.append_trace(go.Scatter(x=df_normal_t.index, y=df_normal_t.iloc[:, 1], mode='lines', line_width=4), row=2,
                     col=1)
    fig.append_trace(go.Scatter(x=df_normal_t.index, y=df_normal_t.iloc[:, 2], mode='lines', line_width=4), row=3,
                     col=1)
    fig.add_vline(x='2004-12', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2008-11', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2012-02', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2014-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2013-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2018-09', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vline(x='2020-03', line_dash="dash", opacity=0.5, line_width=3)
    fig.add_vrect(x0='2003-01', x1='2004-12', annotation_text="ROP", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2003-01', x1='2008-11', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, opacity=0.05)
    fig.add_vrect(x0='2004-12', x1='2012-02', annotation_text="PAHOR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0='2008-11', x1='2013-03', annotation_text="JANSA", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="blue", opacity=0.05)
    fig.add_vrect(x0='2012-02', x1='2014-09', annotation_text="BRATUSEK", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="yellow", opacity=0.05)
    fig.add_vrect(x0='2013-03', x1='2018-09', annotation_text="CERAR", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="pink", opacity=0.05)
    fig.add_vrect(x0='2014-09', x1='2020-03', annotation_text="SAREC", annotation_position="top right",
                  annotation_font_size=30,
                  annotation_textangle=90, line_width=3, fillcolor="black", opacity=0.05)
    # fig.add_vrect(x0='2018-09', x1='2020-05', annotation_text="JANSA", annotation_position="top right", annotation_font_size=30,
    # annotation_textangle=90, line_width=3, fillcolor="orange", opacity=0.05)
    fig.update_layout(yaxis=dict(tickfont=dict(size=30)), yaxis2=dict(tickfont=dict(size=30)),
                      yaxis3=dict(tickfont=dict(size=30)), xaxis1=dict(tickfont=dict(size=30)),
                      xaxis2=dict(tickfont=dict(size=30)), xaxis3=dict(tickfont=dict(size=30)))
    fig.update_layout(height=900, width=2000)
    fig.show()
    fig.write_html('plots/cm/trues.html')


def plot_predictions(config_path):
    df_pred, df_all, biggest = get_data(config_path)

    # lowest_c = lowest['company'].tolist()
    biggest_c = biggest['company'].tolist()
    # middle_c = middle['company'].tolist()

    df_normal = pd.DataFrame()

    for n_company in biggest_c:
        df_icompany = df_all[df_all['podjetje'] == n_company]
        df_normal = pd.concat([df_normal, df_icompany])

    df_normal = df_normal.T
    df_normal.columns = df_normal.iloc[0, :]
    df_normal = df_normal.iloc[1:-1, :]

    fig = make_subplots(rows=3, cols=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 0], mode='lines', line_width=4), row=1, col=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 1], mode='lines', line_width=4), row=2, col=1)
    fig.append_trace(go.Scatter(x=df_normal.index, y=df_normal.iloc[:, 2], mode='lines', line_width=4), row=3, col=1)
    fig.add_vline(x='2004-12', line_dash="dash", opacity=0.5)
    fig.add_vrect(x0='2003-01', x1='2004-12', annotation_text="ROP", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, opacity=0.05)
    fig.add_vrect(x0='2003-01', x1='2008-11', annotation_text="JANSA", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, opacity=0.05)
    fig.add_vline(x='2008-11', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2012-02', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2014-09', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2013-03', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2018-09', line_dash="dash", opacity=0.5)
    fig.add_vline(x='2020-03', line_dash="dash", opacity=0.5)
    fig.add_vrect(x0='2004-12', x1='2012-02', annotation_text="PAHOR", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="green", opacity=0.05)
    fig.add_vrect(x0='2008-11', x1='2013-03', annotation_text="JANSA", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="blue", opacity=0.05)
    fig.add_vrect(x0='2012-02', x1='2014-09', annotation_text="BRATUSEK", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="yellow", opacity=0.05)
    fig.add_vrect(x0='2013-03', x1='2018-09', annotation_text="CERAR", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="pink", opacity=0.05)
    fig.add_vrect(x0='2014-09', x1='2020-03', annotation_text="SAREC", annotation_font_size=30,
                  annotation_position="top right",
                  annotation_textangle=90, line_width=0, fillcolor="black", opacity=0.05)
    fig.update_layout(yaxis=dict(tickfont=dict(size=30)), yaxis2=dict(tickfont=dict(size=30)),
                      yaxis3=dict(tickfont=dict(size=30)), xaxis1=dict(tickfont=dict(size=30)),
                      xaxis2=dict(tickfont=dict(size=30)), xaxis3=dict(tickfont=dict(size=30)))
    fig.update_layout(height=1500, width=1400)
    fig.update_traces(line=dict(width=6))
    fig.show()
    fig.write_html('plots/cm/preds.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    # Figure 6
    plot_cm_examples(args.config)
    # Figure 7
    plot_predictions(args.config)
