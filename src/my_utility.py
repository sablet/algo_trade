from __future__ import print_function
import pandas_datareader.data as web
import csv
import requests
import pandas
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def urls2list(url=None, key="Symbol", kinds='sandp'):
    """
    SandP 500 csv get and collect list
    :param url: str
    :param key: str
    :return: list
    """
    if url is None:
        if kinds is 'sandp':
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    return [item[key] for item
            in csv.DictReader(requests.get(url).text.splitlines())]


def symbols2daily_values(out_fpath='sandp500.h5', key='SandP'):
    """
    return S&P500 stocks daily values during 2010/1/1~2017/1/15
    :param out_fpath: str
    :param key: str
    :return: pandas.Pane;
    """
    if os.path.exists(out_fpath):
        return pandas.read_hdf(out_fpath)
    else:
        name, ext = out_fpath.split('.')
        # default 2010/1/1 ~ today
        print("data collecting...")
        data = web.DataReader(urls2list(), 'yahoo')
        data.to_hdf(out_fpath, key)
        return data


def np1(arr):
    if type(arr) in [pandas.Series, pandas.DataFrame, pandas.Panel]:
        return arr.values.reshape(np.prod(arr.values.shape))
    elif type(arr) is np.ndarray:
        return arr.reshape(np.prod(arr.shape))
    assert False


def double_plot(arr):
    arr = np1(arr)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.distplot(arr)
    plt.subplot(1, 2, 2)
    sns.boxplot(arr)


def daily_values2filtered(pd_panel, key='Adj Close'):
    """
    daily value filtering(key, drop Nan, Change rate
    :param pd_panel: pandas.Panel
    :param key: str
    :return: pandas.DataFrame
    """
    return pd_panel[key].dropna(axis=1)


def df2get_batch(df, term_dict, feature_term=6):
    """
    get batch(feature space and labels)
    :param df: pandas.DataFrame
    :param term_dict: dict
    :param feature_term: int
    :return: numpy.ndarray
    """
    # df = np.log(df.pct_change())
    df = df.pct_change()
    features = {key: np.array([df[:time][-feature_term:].values
                             for time in df[term[0]:term[1]].index])
                               for key, term in term_dict.items()}
    labels = {key: df.shift(1)[term[0]:term[1]].values
                            for key, term in term_dict.items()}
    terms = {key: df[term[0]:term[1]].index for key, term in term_dict.items()}
    return features, labels, terms
