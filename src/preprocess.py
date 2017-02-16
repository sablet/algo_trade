from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA, FastICA


def filter_key_nan(pd_panel, key='Adj Close'):
    """
    daily value filtering(key, drop Nan, Change rate
    :param pd_panel: pandas.Panel
    :param key: str
    :return: pandas.DataFrame
    """
    return np.log(pd_panel[key].dropna(axis=1)).diff()


def panel2get_batch(pd_panel, term_dict, key='Adj Close', feature_term=6):
    """
    get batch(feature space and labels)
    :param key: str
    :param pd_panel: pandas.Panel
    :param term_dict: dict
    :param feature_term: int
    :return: dict
    """
    df = filter_key_nan(pd_panel, key)

    features = {key: np.array([df[:time][-feature_term:].values
                             for time in df[term[0]:term[1]].index])
                               for key, term in term_dict.items()}
    labels = {key: df.shift(-1)[term[0]:term[1]].values
                            for key, term in term_dict.items()}
    terms = {key: df[term[0]:term[1]].index for key, term in term_dict.items()}
    return features, labels, terms


def filter_learn(features, kinds='pca', n_comp=10):
    assert kinds in ['pca', 'ica']
    if kinds is 'pca':
        decomp = PCA(n_components=n_comp)
    elif kinds is 'ica':
        decomp = FastICA(n_components=n_comp)
    else:
        assert False
    decomp.fit(features['train'])
    return {key: decomp.transform(val) for key, val in features.items()}
