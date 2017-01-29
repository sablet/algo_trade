from __future__ import print_function
import pandas_datareader.data as web
import csv
import requests
import pandas
import os
import numpy as np
import json

URL_JSON = 'urllists.json'


def urls2list(url=None, key=None, kind='sandp500'):
    """
    SandP 500 csv get and collect list
    :param kind: str
    :param url: str
    :param key: str
    :return: list
    """
    with open(URL_JSON) as f:
        json_dic = json.load(f)
    if url is None and kind in json_dic.keys():
        url = json_dic[kind]
    if key is None:
        return [item for item in csv.DictReader(
             requests.get(url).text.splitlines())]
    else:
        return [item[key] for item in csv.DictReader(
             requests.get(url).text.splitlines())]


def symbols2daily_values(kinds='sandp500'):
    """
    return S&P500 stocks daily values during 2010/1/1~2017/1/15
    :param kinds: str
    :return: pandas.Pane;
    """
    out_fpath = kinds + '.h5'
    if os.path.exists(out_fpath):
        return pandas.read_hdf(out_fpath)
    else:
        print("data collecting...")
        data = web.DataReader(urls2list(key='Symbol'), 'yahoo')
        data.to_hdf(out_fpath, kinds)
        return data


def np1(arr):
    if type(arr) in [pandas.Series, pandas.DataFrame, pandas.Panel]:
        return arr.values.reshape(np.prod(arr.values.shape))
    elif type(arr) is np.ndarray:
        return arr.reshape(np.prod(arr.shape))
    assert False


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
    features = {key: np.array([df[:time][-feature_term:].values
                             for time in df[term[0]:term[1]].index])
                               for key, term in term_dict.items()}
    labels = {key: df.shift(1)[term[0]:term[1]].values
                            for key, term in term_dict.items()}
    terms = {key: df[term[0]:term[1]].index for key, term in term_dict.items()}
    return features, labels, terms
