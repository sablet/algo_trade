import numpy as np
import pandas
import os
import subprocess


def np1(data):
    if type(data) in [pandas.Series, pandas.DataFrame, pandas.Panel]:
        return data.values.reshape(np.prod(data.values.shape))
    elif type(data) is np.ndarray:
        return data.reshape(np.prod(data.shape))
    elif type(data) is dict:
        return {key: val.reshape(np.prod(val.shape)) for key, val in data.items()}
    assert False


def np3to2(data):
    if type(data) is np.ndarray:
        shape = data.shape
        return data.reshape(shape[0], shape[1] * shape[2])
    elif type(data) is dict:
        return {key: val.reshape(val.shape[0], val.shape[1] * val.shape[2])
                for key, val in data.items()}
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
