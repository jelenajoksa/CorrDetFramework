import pandas as pd
import argparse
import yaml
import cutoff
from cutoff import read_params
from cutoff import get_data
from cutoff import cutoff
import datetime
import time
from datetime import timedelta
from datetime import date
from dateutil.relativedelta import relativedelta



def get_event(config_path):
    config = read_params(config_path)
    gov_year = config['event']['year']
    gov_month = config['event']['month']
    gov_day = config['event']['day']
    gov_date = date(gov_year,gov_month,gov_day )
    period = config['period']
    cutoff_data_path = config['processed_data']['cutoff']
    df = pd.read_csv(cutoff_data_path, sep=',')
    #print(df.head())
    d1 = gov_date - relativedelta(months = period)
    d2 = gov_date + relativedelta(months = period)
    year1 = d1.strftime("%Y")
    month1 = d1.strftime("%m")
    year2 = d2.strftime("%Y")
    month2 = d2.strftime("%m")
    date1 = str(year1) + '-' + str(month1)
    date2 = str(year2) + '-' + str(month2)
    gov_df = df.loc[:, date1:date2]
    gov_df['podjetje'] = df['podjetje']
    #remove companies that had only 0eur income in this particular period
    gov_df = gov_df.loc[~(gov_df.loc[:, date1:date2] == 0).all(axis=1)]
    gov_data_path = config['processed_data']['event_data']
    gov_df.to_csv(gov_data_path, sep=",", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")

    parser.add_argument("--config", default="params.yaml")

    args = parser.parse_args()
    get_event(args.config)




