from __future__ import print_function
import pandas_datareader.data as web
import csv
import os
import subprocess
import requests
import pandas


def get_data(input_fpath='sandp500.h5', key='SandP'):
    """
    return S&P500 stocks daily values during 2010/1/1~2017/1/15
    :param input_fpath: str
    :param key: str
    :return: pandas.DataFrame
    """
    if os.path.exists(input_fpath):
        return pandas.read_hdf(input_fpath)
    else:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        symbold_list = requests.get(url).text.splitlines()
        name, ext = input_fpath.split('.')
        # default 2010/1/1 ~ today
        print("data collecting...")
        data = web.DataReader(
            [item['Symbol'] for item in csv.DictReader(symbold_list)],
            'yahoo'
        )
        data.to_hdf(input_fpath, key)
        return data


def df2(pd_data):
    # key, term, dropna adopt
    input_d = pd_data['Adj Close']['2010-12-20':].dropna(axis=1).pct_change()
