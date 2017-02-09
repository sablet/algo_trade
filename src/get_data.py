import csv
import json
import os
import pandas
import requests
from pandas_datareader import data as web
from src.utility import get_out_path, get_root
from YahooJapanDataReader.io.data import DataReader

URL_JSON = 'urllists.json'


def csvurls2list(url=None, key=None, kind='sandp500'):
    """
    SandP 500 csv get and collect list
    :param kind: str
    :param url: str
    :param key: str
    :return: list
    """
    with open(get_out_path(URL_JSON)) as f:
        json_dic = json.load(f)
    if url is None and kind in json_dic.keys():
        url = json_dic[kind]
    if key is None:
        return [item for item in csv.DictReader(
            requests.get(url).text.splitlines())]
    elif key is 'nikkei255':
        return list(
            map(
                int,
                pandas.io.html.read_html(
                    'http://swing-trade.net/nk225itiran')[0]
            ))
    else:
        return [item[key] for item in csv.DictReader(
            requests.get(url).text.splitlines())]


def symbols2daily_values(kinds='sandp500', symbols=None):
    """
    return S&P500 stocks daily values during 2010/1/1~2017/1/15
    :param symbols: List[str]
    :param kinds: str
    :return: pandas.Pane
    """
    out_path = get_out_path(kinds + '.h5')
    if not os.path.exists(out_path):
        if kinds is 'nikkei225':
            stock_dic = {}
            for stock_code in list(map(int, csvurls2list(kind=kinds)['コード'])):
                stock_dic[stock_code] = DataReader(
                    stock_code,
                    data_source='yahoojp',
                    start='2010-01-01',
                    end='2016-12-31',
                    adjust=True
                )
            pandas.Panel(
                {key: value.reset_index() for key, value in stock_dic.items()})\
                .swapaxes('items', 'minor')\
                [['Close', 'High', 'Low', 'Open', 'Volume']]\
                .astype('float64')\
                .to_hdf(out_path, kinds)
        else:
            print("data collecting...")
            if symbols is None:
                symbols = csvurls2list(key='Symbol')
            web.DataReader(symbols, 'yahoo').to_hdf(out_path, kinds)
    return pandas.read_hdf(out_path, kinds)
