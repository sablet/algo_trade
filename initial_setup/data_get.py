from __future__ import print_function
import pandas_datareader.data as web
import requests_cache
import csv
import os


FILE_NAME = "data/sandp500_lists.csv"

assert os.path.exists(FILE_NAME)
with open(FILE_NAME) as f_obj:
    # default 2010/1/1 ~ today
    print("data collecting...")
    stocks = web.DataReader(
        [item['Symbol'] for item in csv.DictReader(f_obj)],
        'yahoo',
        session=requests_cache.CachedSession(
                cache_name='sandp500',
                backend='sqlite'
        )
    )
