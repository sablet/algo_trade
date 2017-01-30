import csv
import json
import os
import pandas
import requests
from pandas_datareader import data as web

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
    :return: pandas.Pane
    """
    out_fpath = kinds + '.h5'
    if os.path.exists(out_fpath):
        return pandas.read_hdf(out_fpath)
    else:
        print("data collecting...")
        data = web.DataReader(urls2list(key='Symbol'), 'yahoo')
        data.to_hdf(out_fpath, kinds)
        return data
