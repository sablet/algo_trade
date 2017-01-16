from __future__ import print_function
import pandas_datareader.data as web
import csv
import requests
import pandas
import os


def sandp_url2symbols(url=None, key="Symbol"):
    """
    SandP 500 csv get and collect list
    :param url: str
    :param key: str
    :return: list
    """
    if url is None:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    return [item[key] for item
            in csv.DictReader(requests.get(url).text.splitlines())]


def symbols2daily_values(symbol_list, out_fpath='sandp500.h5', key='SandP'):
    """
    return S&P500 stocks daily values during 2010/1/1~2017/1/15
    :param symbol_list: List[str]
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
        data = web.DataReader(symbol_list, 'yahoo')
        data.to_hdf(out_fpath, key)
        return data


def daily_values2filterd(pd_panel, key='Adj Close'):
    """
    daily value filtering(key, drop Nan, Change rate
    :param pd_panel: pandas.Panel
    :param key: str
    :return: pandas.DataFrame
    """
    return pd_panel[key].dropna(axis=1).pct_change()


def values2get_batch(values, term_dict, feature_term=6):
    """
    get batch(feature space and labels)
    :param values: pandas.DataFrame
    :param term_dict: dict
    :param feature_term: int
    :return: numpy.ndarray
    """
    features = {key: values[:time][-feature_term:].values
                for key, term in term_dict.items()
                for time in values[term[0]:term[1]].index}
    labels = {key: (values.shift(-1))[:].ix[time].values
              for key, term in term_dict.items()
              for time in values[term[0]:term[1]].index}
    return features, labels


def foldl_apply(f_list):
    """
    left foldable
    :param f_list: List[func]
    :return:
    """
    if len(f_list) == 1:
        return f_list[0]()
    else:
        return f_list[0](foldl_apply(f_list[1:]))
