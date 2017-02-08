import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utility import np1
from src.get_data import urls2list, symbols2daily_values
from src.preprocess import


def df_violin(df, linewidth=4.0):
    assert type(df) is pd.DataFrame
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    sns.violinplot(df.mean(axis=0), linewidth=linewidth)
    plt.xlabel('$E_t[c_{t,i}]$', fontsize=16)
    plt.subplot(222)
    sns.violinplot(df.std(axis=0), linewidth=linewidth)
    plt.xlabel('$\sigma_t[c_{t,i}]$', fontsize=16)
    plt.subplot(223)
    sns.violinplot(df.mean(axis=1), linewidth=linewidth)
    plt.xlabel('$E_i[c_{t,i}]$', fontsize=16)
    plt.subplot(224)
    sns.violinplot(df.std(axis=1), linewidth=linewidth)
    plt.xlabel('$\sigma_i[c_{t,i}]$', fontsize=16)


def df_describe(df):
    assert type(df) is pd.DataFrame
    pd.DataFrame(np1(df), columns=['whole distribution']).describe()
