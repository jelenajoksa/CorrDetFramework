import pandas as pd
import argparse
import yaml

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config['data_source']['source']
    df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    return df

def cutoff(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    df['sum'] = df.sum(axis=1, numeric_only=True)
    df = df[(df['sum'] > config["cutoff"]["onemill"])]
    cutoff_data_path = config["processed_data"]["cutoff"]
    df.to_csv(cutoff_data_path, sep=",", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    get_data(args.config)
    cutoff(args.config)
