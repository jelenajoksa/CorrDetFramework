import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import silhouette_score


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path_event = config['processed_data']['event_data']
    df_event = pd.read_csv(data_path_event, sep=',', encoding='utf-8')
    data_path_cutoff = config['processed_data']['cutoff']
    df_cutoff = pd.read_csv(data_path_cutoff, sep=',', encoding='utf-8')
    return df_event, df_cutoff


def clustering(config_path):
    config = read_params(config_path)
    df, whole_df = get_data(config_path)
    df_num = df.iloc[:, :-1]
    arr_df = df_num.to_numpy()
    X_train = TimeSeriesScalerMeanVariance().fit_transform(arr_df)
    no_clusters = config['clustering']['no_clusters']
    algo = TimeSeriesKMeans(n_clusters=no_clusters, verbose=True, random_state=config['clustering']['random_state'])
    y_pred = algo.fit_predict(X_train)
    n, m, k = X_train.shape
    arr_X_train = np.reshape(X_train, (n, m))
    silhouette_avg = silhouette_score(arr_X_train, y_pred)
    labels = algo.labels_
    companies = df.iloc[:, -1]
    #KMeans + Euclidean
    algo = TimeSeriesKMeans(n_clusters = no_clusters, verbose=True, random_state=10)
    y_pred = algo.fit_predict(X_train)
    #add received labels to cutoff dataset
    whole_df = whole_df.iloc[:, :-1]
    df_num['podjetje'] = companies
    df_num['cluster'] = labels
    df_num_cut = df_num.iloc[:, -2:]
    merge_df = pd.merge(df_num_cut, whole_df, on='podjetje')
    clusters_df = merge_df.iloc[:, -211:]
    clustered_data_path = config['processed_data']['clustered_data']
    clusters_df.to_csv(clustered_data_path, sep=",", index=False)
    return algo, X_train, y_pred, silhouette_avg

#Output1: Plots

#We plot 10 separate figures with 10 clusters in each, for the better preview
def km_plot(config_path):
    algo, X_train, y_pred, silhouette_avg = clustering(config_path)
    for i in range(1,4):
        plt.figure()
        for yi in range((i-1)*10,(i-1)*10+10):
                plt.subplot(2,5,yi-(i-1)*10+1)
                for xx in X_train[y_pred == yi]:
                    plt.plot(xx.ravel(), "k-", alpha=.05)
                plt.plot(algo.cluster_centers_[yi].ravel(), "r-")
                plt.axvline(x=6, color='b', ls='--', lw=1)
                plt.text(-3, -3, 'Cluster %d' % (yi))
                if yi == 1:
                   plt.title("Euclidean $k$-means (MeanVar)\n"+str(round(silhouette_avg,3)))
        plt.savefig('plots/kmeans/event_2014/clusters_{}.png'.format(i))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    clustering(args.config)
    km_plot(args.config)
