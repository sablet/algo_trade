import numpy as np
import pandas
import os
import subprocess


def np1(arr):
    if type(arr) in [pandas.Series, pandas.DataFrame, pandas.Panel]:
        return arr.values.reshape(np.prod(arr.values.shape))
    elif type(arr) is np.ndarray:
        return arr.reshape(np.prod(arr.shape))
    assert False


def np3to2(arr):
    if type(arr) is np.ndarray:
        shape = arr.shape
        return arr.reshape(shape[0], shape[1] * shape[2])
    assert False


def get_root():
    return subprocess.getoutput("git rev-parse --show-toplevel")


def move2root():
    return os.chdir(get_root())


def get_out_path(file_name, dname='data'):
    os.makedirs(
        os.path.join(get_root(), dname),
        exist_ok=True
    )
    return os.path.join(get_root(), dname, file_name)
